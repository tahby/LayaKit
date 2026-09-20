import CoreML
import XCTest
@testable import LayaKit

enum SharedAgent {
    private static nonisolated(unsafe) var cached: LayaAgent?

    static var bundlePath: String {
        ProcessInfo.processInfo.environment["LAYA_GENERAL_BUNDLE"] ?? defaultBundlePath
    }

    private static var defaultBundlePath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/general")
            .path
    }

    static func load() async throws -> LayaAgent {
        if let cached { return cached }
        let agent = try await LayaAgent(bundle: URL(fileURLWithPath: bundlePath))
        cached = agent
        return agent
    }
}

struct FixtureCase {
    let id: String
    let state: String
    let question: LayaQuestion
    let tokenIds: [Int]
    let markers: [Int]
    let qtype: Int
    let collated: OrderedJSON
    let logits: [Float]
    let actionLogits: [Float]
    let answer: OrderedJSON
}

func loadFixtures() -> (special: OrderedJSON, cases: [FixtureCase]) {
    let url = Bundle.module.url(forResource: "fixtures", withExtension: "json")!
    let root = OrderedJSON.parse(try! Data(contentsOf: url))
    let cases = root["cases"]!.arrayValue!.map { item -> FixtureCase in
        let general = item["general"]!
        return FixtureCase(
            id: item["id"]!.stringValue!,
            state: item["state"]!.stringValue!,
            question: decodeQuestion(item["question"]!),
            tokenIds: item["token_ids"]!.arrayValue!.map { $0.intValue! },
            markers: item["markers"]!.arrayValue!.map { $0.intValue! },
            qtype: item["qtype"]!.intValue!,
            collated: general["collated"]!,
            logits: general["logits"]!.arrayValue![0].arrayValue!.map { Float($0.doubleValue!) },
            actionLogits: general["action_logits"]!.arrayValue![0].arrayValue!.map { Float($0.doubleValue!) },
            answer: general["answer"]!
        )
    }
    return (root["special_tokens"]!, cases)
}

func decodeQuestion(_ json: OrderedJSON) -> LayaQuestion {
    let instructions = json["instructions"]!.stringValue!
    let criteria = json["criteria"]!
    switch json["type"]!.stringValue! {
    case "choice":
        if let list = criteria.arrayValue {
            return .choice(instructions: instructions, options: list.map { $0.stringValue! })
        }
        let entries = criteria.entries!
        var descriptions: [String: String] = [:]
        for entry in entries where !entry.value.isNull {
            descriptions[entry.key] = entry.value.stringValue!
        }
        return .choice(instructions: instructions, options: entries.map(\.key), descriptions: descriptions)
    case "score":
        return .score(instructions: instructions, levels: criteria.arrayValue!.map { $0.stringValue! })
    default:
        let entries = criteria.entries
        return .noul(
            instructions: instructions,
            falseText: entries?.first { $0.key == "false" }?.value.stringValue,
            trueText: entries?.first { $0.key == "true" }?.value.stringValue
        )
    }
}

final class LayaKitTests: XCTestCase {
    let fixtures = loadFixtures()

    override func setUp() async throws {
        guard FileManager.default.fileExists(atPath: SharedAgent.bundlePath) else {
            throw XCTSkip("General model bundle not found at \(SharedAgent.bundlePath)")
        }
    }

    func testSpecialTokenIds() async throws {
        let agent = try await SharedAgent.load()
        let special = agent.promptBuilder.tokenizer.special
        XCTAssertEqual(special.cls, fixtures.special["cls"]!.intValue!)
        XCTAssertEqual(special.sep, fixtures.special["sep"]!.intValue!)
        XCTAssertEqual(special.pad, fixtures.special["pad"]!.intValue!)
        XCTAssertEqual(special.mask, fixtures.special["mask"]!.intValue!)
    }

    func testTokenizationMatchesFixtures() async throws {
        let agent = try await SharedAgent.load()
        for fixture in fixtures.cases {
            let normalized = try NormalizedQuestion(fixture.question)
            let item = try agent.promptBuilder.build(state: fixture.state, question: normalized)
            XCTAssertEqual(item.ids, fixture.tokenIds, fixture.id)
            XCTAssertEqual(item.markers, fixture.markers, fixture.id)
            XCTAssertEqual(Int(item.qtype.rawValue), fixture.qtype, fixture.id)
        }
    }

    func testCollationMatchesFixtures() async throws {
        let agent = try await SharedAgent.load()
        for fixture in fixtures.cases {
            let normalized = try NormalizedQuestion(fixture.question)
            let batch = try agent.collator.collate(
                agent.promptBuilder.build(state: fixture.state, question: normalized)
            )
            let expected = fixture.collated
            XCTAssertEqual(batch.length, expected["padded_length"]!.intValue!, fixture.id)
            XCTAssertEqual(batch.inputIds, row(expected, "input_ids"), fixture.id)
            XCTAssertEqual(batch.attentionMask, row(expected, "attention_mask"), fixture.id)
            XCTAssertEqual(batch.markerPos, row(expected, "marker_pos"), fixture.id)
            XCTAssertEqual(batch.markerMask, row(expected, "marker_mask"), fixture.id)
            XCTAssertEqual(batch.qtype, expected["qtype"]!.arrayValue!.map { Int32($0.intValue!) }, fixture.id)
        }
    }

    func testRawLogitsMatchFixtures() async throws {
        let agent = try await SharedAgent.load()
        for fixture in fixtures.cases {
            let normalized = try NormalizedQuestion(fixture.question)
            let batch = try agent.collator.collate(
                agent.promptBuilder.build(state: fixture.state, question: normalized)
            )
            let (logits, actionLogits) = try agent.backend.forward(batch)
            for index in 0..<batch.optionCount {
                XCTAssertEqual(
                    Double(logits[index]), Double(fixture.logits[index]), accuracy: 0.05,
                    "\(fixture.id) logit \(index)"
                )
            }
            for index in 0..<2 {
                let expected = Double(fixture.actionLogits[index])
                XCTAssertEqual(
                    Double(actionLogits[index]), expected, accuracy: abs(expected) * 0.01,
                    "\(fixture.id) action logit \(index)"
                )
            }
        }
    }

    func testAnswersMatchFixtures() async throws {
        let agent = try await SharedAgent.load()
        for fixture in fixtures.cases {
            let answer = try agent.predict(state: fixture.state, question: fixture.question)
            let expected = fixture.answer
            XCTAssertEqual(
                answer.confidence, expected["confidence"]!.doubleValue!, accuracy: 0.005, fixture.id
            )
            XCTAssertEqual(
                answer.actProbability,
                expected["action"]!["act_probability"]!.doubleValue!,
                accuracy: 0.005, fixture.id
            )
            switch expected["type"]!.stringValue! {
            case "choice":
                XCTAssertEqual(answer.choice, expected["choice"]!.stringValue!, fixture.id)
                let probabilities = expected["probabilities"]!.entries!
                XCTAssertEqual(answer.labels, probabilities.map(\.key), fixture.id)
                for (index, entry) in probabilities.enumerated() {
                    XCTAssertEqual(
                        answer.probabilities[index], entry.value.doubleValue!, accuracy: 0.005,
                        "\(fixture.id) \(entry.key)"
                    )
                }
            case "score":
                XCTAssertEqual(answer.score!, expected["score"]!.doubleValue!, accuracy: 0.005, fixture.id)
                XCTAssertEqual(answer.labels, expected["legend"]!.entries!.map { $0.value.stringValue! }, fixture.id)
                for (index, entry) in expected["probabilities"]!.entries!.enumerated() {
                    XCTAssertEqual(
                        answer.probabilities[index], entry.value.doubleValue!, accuracy: 0.005,
                        "\(fixture.id) \(entry.key)"
                    )
                }
            default:
                XCTAssertEqual(answer.noul!, expected["noul"]!.doubleValue!, accuracy: 0.005, fixture.id)
                XCTAssertEqual(answer.labels, ["false", "true"], fixture.id)
            }
        }
    }

    func testPredictLatency() async throws {
        let agent = try await SharedAgent.load()
        let fixture = fixtures.cases[0]
        _ = try agent.predict(state: fixture.state, question: fixture.question)
        let start = Date()
        let runs = 20
        for _ in 0..<runs {
            _ = try agent.predict(state: fixture.state, question: fixture.question)
        }
        let averageSeconds = Date().timeIntervalSince(start) / Double(runs)
        XCTAssertLessThan(averageSeconds, 0.5)
    }

    func testDuplicateChoiceLabelsRejected() {
        assertInvalid(.choice(instructions: "pick", options: ["a", "a"]))
    }

    func testEmptyChoiceOptionsRejected() {
        assertInvalid(.choice(instructions: "pick", options: []))
    }

    func testEmptyScoreLevelsRejected() {
        assertInvalid(.score(instructions: "rate", levels: []))
    }

    func testTooManyOptionsRejected() async throws {
        let agent = try await SharedAgent.load()
        let options = (0..<33).map { "option\($0)" }
        XCTAssertThrowsError(
            try agent.predict(state: "anything", question: .choice(instructions: "pick", options: options))
        ) { error in
            guard case let LayaError.tooManyOptions(count, limit) = error else {
                return XCTFail("expected tooManyOptions, got \(error)")
            }
            XCTAssertEqual(count, 33)
            XCTAssertEqual(limit, 32)
        }
    }

    func testTooManyTokensRejected() async throws {
        let agent = try await SharedAgent.load()
        let item = TokenizedItem(ids: Array(repeating: 5, count: 1025), markers: [1], qtype: .choice)
        XCTAssertThrowsError(try agent.collator.collate(item)) { error in
            guard case let LayaError.tooManyTokens(count, limit) = error else {
                return XCTFail("expected tooManyTokens, got \(error)")
            }
            XCTAssertEqual(count, 1025)
            XCTAssertEqual(limit, 1024)
        }
    }

    func testAneBundleRejected() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = #"{"format":"laya-coreml-ane","shape":{"batch_size":1,"max_length":1024,"min_length":16,"max_options":32,"flexible":true,"lengths":[16]}}"#
        try manifest.write(
            to: directory.appendingPathComponent("coreml_config.json"), atomically: true, encoding: .utf8
        )
        do {
            _ = try await LayaAgent(bundle: directory)
            XCTFail("expected unsupportedBundle")
        } catch let LayaError.unsupportedBundle(format) {
            XCTAssertEqual(format, "laya-coreml-ane")
        }
    }

    private func assertInvalid(_ question: LayaQuestion) {
        XCTAssertThrowsError(try NormalizedQuestion(question)) { error in
            guard case LayaError.invalidQuestion = error else {
                return XCTFail("expected invalidQuestion, got \(error)")
            }
        }
    }

    private func row(_ collated: OrderedJSON, _ name: String) -> [Int32] {
        collated[name]!.arrayValue![0].arrayValue!.map { Int32($0.intValue!) }
    }
}
