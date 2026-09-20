import ArgumentParser
import Foundation
import LayaKit

enum CLIError: Error, CustomStringConvertible {
    case invalidQuestionFile(String)

    var description: String {
        switch self {
        case let .invalidQuestionFile(reason): return "Invalid question file: \(reason)"
        }
    }
}

func loadQuestion(path: String) throws -> (LayaQuestion, String) {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let json = try OrderedJSON.parse(data)
    guard let type = json["type"]?.stringValue else {
        throw CLIError.invalidQuestionFile("missing \"type\"")
    }
    guard let instructions = json["instructions"]?.stringValue else {
        throw CLIError.invalidQuestionFile("missing \"instructions\"")
    }
    let criteria = json["criteria"]
    switch type {
    case "choice":
        if let entries = criteria?.entries {
            let options = entries.map { $0.key }
            var descriptions: [String: String] = [:]
            for entry in entries {
                if let text = entry.value.stringValue {
                    descriptions[entry.key] = text
                }
            }
            return (.choice(instructions: instructions, options: options, descriptions: descriptions), type)
        } else if let array = criteria?.arrayValue {
            let options = array.compactMap { $0.stringValue }
            return (.choice(instructions: instructions, options: options), type)
        } else {
            throw CLIError.invalidQuestionFile("\"criteria\" must be a list or object for a choice question")
        }
    case "score":
        guard let array = criteria?.arrayValue else {
            throw CLIError.invalidQuestionFile("\"criteria\" must be a list for a score question")
        }
        let levels = array.compactMap { $0.stringValue }
        return (.score(instructions: instructions, levels: levels), type)
    case "noul":
        if let criteria, criteria.entries == nil {
            throw CLIError.invalidQuestionFile("\"criteria\" must be an object with \"true\"/\"false\" keys for a noul question")
        }
        let falseText = criteria?["false"]?.stringValue
        let trueText = criteria?["true"]?.stringValue
        return (.noul(instructions: instructions, falseText: falseText, trueText: trueText), type)
    default:
        throw CLIError.invalidQuestionFile("unknown \"type\" \(type)")
    }
}

func formatRounded(_ value: Double) -> String {
    let rounded = (value * 10000).rounded() / 10000
    var text = String(format: "%.4f", rounded)
    while text.hasSuffix("0") {
        text.removeLast()
    }
    if text.hasSuffix(".") {
        text.append("0")
    }
    return text
}

func escapeJSON(_ text: String) -> String {
    var result = ""
    for scalar in text.unicodeScalars {
        switch scalar {
        case "\"": result += "\\\""
        case "\\": result += "\\\\"
        case "\n": result += "\\n"
        case "\t": result += "\\t"
        case "\r": result += "\\r"
        default: result.unicodeScalars.append(scalar)
        }
    }
    return result
}

indirect enum OutputValue {
    case string(String)
    case rawNumber(String)
    case object([(String, OutputValue)])
}

func render(_ value: OutputValue, indent: Int) -> String {
    let pad = String(repeating: "  ", count: indent)
    let childPad = String(repeating: "  ", count: indent + 1)
    switch value {
    case let .string(text):
        return "\"\(escapeJSON(text))\""
    case let .rawNumber(text):
        return text
    case let .object(entries):
        if entries.isEmpty { return "{}" }
        let lines = entries.map { key, val in
            "\(childPad)\"\(escapeJSON(key))\": \(render(val, indent: indent + 1))"
        }
        return "{\n\(lines.joined(separator: ",\n"))\n\(pad)}"
    }
}

func buildOutput(answer: LayaAnswer, questionType: String) -> String {
    var entries: [(String, OutputValue)] = [
        ("type", .string(questionType)),
        ("confidence", .rawNumber(formatRounded(answer.confidence))),
        ("action", .object([("act_probability", .rawNumber(formatRounded(answer.actProbability)))]))
    ]
    switch questionType {
    case "choice":
        entries.append(("choice", .string(answer.choice ?? "")))
        entries.append((
            "probabilities",
            .object(zip(answer.labels, answer.probabilities).map { ($0, .rawNumber(formatRounded($1))) })
        ))
    case "score":
        entries.append(("score", .rawNumber(formatRounded(answer.score ?? 0))))
        entries.append((
            "legend",
            .object(answer.labels.enumerated().map { (String($0.offset), .string($0.element)) })
        ))
        entries.append((
            "probabilities",
            .object(answer.probabilities.enumerated().map { (String($0.offset), .rawNumber(formatRounded($0.element))) })
        ))
    default:
        entries.append(("noul", .rawNumber(formatRounded(answer.noul ?? 0))))
    }
    return render(.object(entries), indent: 0)
}

func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    let position = max(0, min(sorted.count - 1, Int((fraction * Double(sorted.count)).rounded(.up)) - 1))
    return sorted[position]
}

@main
struct LayaCLI: AsyncParsableCommand {
    @Option(name: .long) var bundle: String
    @Option(name: .long) var state: String
    @Option(name: .long) var question: String
    @Option(name: .long) var bench: Int = 0

    func run() async throws {
        do {
            try await execute()
        } catch {
            FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
            Foundation.exit(1)
        }
    }

    func execute() async throws {
        let (layaQuestion, questionType) = try loadQuestion(path: question)
        let agent = try await LayaAgent(bundle: URL(fileURLWithPath: bundle))
        let answer = try agent.predict(state: state, question: layaQuestion)
        print(buildOutput(answer: answer, questionType: questionType))

        if bench > 0 {
            var latencies: [Double] = []
            latencies.reserveCapacity(bench)
            for _ in 0..<bench {
                let start = DispatchTime.now()
                _ = try agent.predict(state: state, question: layaQuestion)
                let end = DispatchTime.now()
                latencies.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
            }
            latencies.sort()
            let mean = latencies.reduce(0, +) / Double(latencies.count)
            let p50 = percentile(latencies, 0.5)
            let p90 = percentile(latencies, 0.9)
            let message = "p50: \(String(format: "%.3f", p50)) ms, p90: \(String(format: "%.3f", p90)) ms, mean: \(String(format: "%.3f", mean)) ms\n"
            FileHandle.standardError.write(message.data(using: .utf8)!)
        }
    }
}
