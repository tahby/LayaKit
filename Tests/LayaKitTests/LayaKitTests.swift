import CoreML
import XCTest
@testable import LayaKit

private struct ModelBundleMissing: Error {}

func requireModelBundle(_ path: String, envVar: String, repository: String) throws {
    guard !FileManager.default.fileExists(atPath: path) else { return }
    guard ProcessInfo.processInfo.environment["LAYA_SKIP_MODEL_TESTS"] == "1" else {
        XCTFail(
            "\(envVar) bundle not found at \(path). Set \(envVar) to its location, or download it: "
            + "hf download \(repository) --local-dir \(path). Set LAYA_SKIP_MODEL_TESTS=1 to skip these tests."
        )
        throw ModelBundleMissing()
    }
    throw XCTSkip("\(envVar) bundle not found at \(path); LAYA_SKIP_MODEL_TESTS=1 set")
}

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

enum SharedANEAgent {
    private static nonisolated(unsafe) var cached: LayaAgent?

    static var bundlePath: String {
        ProcessInfo.processInfo.environment["LAYA_ANE_BUNDLE"] ?? defaultBundlePath
    }

    private static var defaultBundlePath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/ane")
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
    let exceedsAne: Bool
    let aneCollated: OrderedJSON?
    let aneLogits: [Float]?
    let aneActionLogits: [Float]?
    let aneAnswer: OrderedJSON?
}

func loadFixtures() -> (special: OrderedJSON, cases: [FixtureCase]) {
    let url = Bundle.module.url(forResource: "fixtures", withExtension: "json")!
    let root = try! OrderedJSON.parse(try! Data(contentsOf: url))
    let cases = root["cases"]!.arrayValue!.map { item -> FixtureCase in
        let general = item["general"]!
        let ane = item["ane"]
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
            answer: general["answer"]!,
            exceedsAne: item["exceeds_ane"]?.boolValue ?? false,
            aneCollated: ane?["collated"],
            aneLogits: ane?["logits"]?.arrayValue?[0].arrayValue?.map { Float($0.doubleValue!) },
            aneActionLogits: ane?["action_logits"]?.arrayValue?[0].arrayValue?.map { Float($0.doubleValue!) },
            aneAnswer: ane?["answer"]
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
        try requireModelBundle(
            SharedAgent.bundlePath, envVar: "LAYA_GENERAL_BUNDLE", repository: "aac6fef/laya-multilingual-coreml"
        )
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
        var lastAnswer: LayaAnswer?
        for _ in 0..<runs {
            lastAnswer = try agent.predict(state: fixture.state, question: fixture.question)
        }
        let averageSeconds = Date().timeIntervalSince(start) / Double(runs)
        print("general backend average predict latency: \(averageSeconds)s over \(runs) runs")
        XCTAssertTrue(averageSeconds.isFinite)
        XCTAssertTrue(lastAnswer?.confidence.isFinite ?? false)
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

    func testUnknownFormatRejected() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = #"{"format":"laya-coreml-other","shape":{"batch_size":1,"max_length":1024,"min_length":16,"max_options":32,"flexible":true,"lengths":[16]}}"#
        try manifest.write(
            to: directory.appendingPathComponent("coreml_config.json"), atomically: true, encoding: .utf8
        )
        do {
            _ = try await LayaAgent(bundle: directory)
            XCTFail("expected unsupportedBundle")
        } catch let LayaError.unsupportedBundle(format) {
            XCTAssertEqual(format, "laya-coreml-other")
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

final class ANEBackendTests: XCTestCase {
    let fixtures = loadFixtures()

    override func setUp() async throws {
        try requireModelBundle(
            SharedANEAgent.bundlePath, envVar: "LAYA_ANE_BUNDLE", repository: "aac6fef/laya-multilingual-coreml-ane"
        )
    }

    func testAneCollationMatchesFixtures() async throws {
        let agent = try await SharedANEAgent.load()
        for fixture in fixtures.cases {
            guard let expected = fixture.aneCollated else { continue }
            let normalized = try NormalizedQuestion(fixture.question)
            let batch = try agent.collator.collate(
                agent.promptBuilder.build(state: fixture.state, question: normalized)
            )
            XCTAssertEqual(batch.length, expected["padded_length"]!.intValue!, fixture.id)
            XCTAssertEqual(batch.inputIds, row(expected, "input_ids"), fixture.id)
            XCTAssertEqual(batch.attentionMask, row(expected, "attention_mask"), fixture.id)
            XCTAssertEqual(batch.markerPos, row(expected, "marker_pos"), fixture.id)
            XCTAssertEqual(batch.markerMask, row(expected, "marker_mask"), fixture.id)
            XCTAssertEqual(batch.qtype, expected["qtype"]!.arrayValue!.map { Int32($0.intValue!) }, fixture.id)
        }
    }

    func testAneRawLogitsMatchFixtures() async throws {
        let agent = try await SharedANEAgent.load()
        for fixture in fixtures.cases {
            guard let expectedLogits = fixture.aneLogits, let expectedAction = fixture.aneActionLogits else { continue }
            let normalized = try NormalizedQuestion(fixture.question)
            let batch = try agent.collator.collate(
                agent.promptBuilder.build(state: fixture.state, question: normalized)
            )
            let (logits, actionLogits) = try agent.backend.forward(batch)
            for index in 0..<batch.optionCount {
                XCTAssertEqual(
                    Double(logits[index]), Double(expectedLogits[index]), accuracy: 0.05,
                    "\(fixture.id) logit \(index)"
                )
            }
            for index in 0..<2 {
                let expected = Double(expectedAction[index])
                XCTAssertEqual(
                    Double(actionLogits[index]), expected, accuracy: abs(expected) * 0.01,
                    "\(fixture.id) action logit \(index)"
                )
            }
        }
    }

    func testAneAnswersMatchFixtures() async throws {
        let agent = try await SharedANEAgent.load()
        for fixture in fixtures.cases {
            guard let expected = fixture.aneAnswer else { continue }
            let answer = try agent.predict(state: fixture.state, question: fixture.question)
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

    func testAneExceedsMaxLengthThrows() async throws {
        let agent = try await SharedANEAgent.load()
        guard let fixture = fixtures.cases.first(where: { $0.exceedsAne }) else {
            return XCTFail("expected an exceeds_ane fixture case")
        }
        XCTAssertThrowsError(try agent.predict(state: fixture.state, question: fixture.question)) { error in
            guard case LayaError.tooManyTokens = error else {
                return XCTFail("expected tooManyTokens, got \(error)")
            }
        }
    }

    func testSafetensorsFileReadsHostWeights() throws {
        let url = URL(fileURLWithPath: SharedANEAgent.bundlePath).appendingPathComponent("host_weights.safetensors")
        let file = try SafetensorsFile(url: url)

        let embedding = try file.entry("encoder.embeddings.tok_embeddings.weight")
        XCTAssertEqual(embedding.dtype, "F16")
        XCTAssertEqual(embedding.shape, [256000, 768])

        let typeEmb = try file.entry("type_emb.weight")
        XCTAssertEqual(typeEmb.dtype, "F16")
        XCTAssertEqual(typeEmb.shape, [3, 768])

        let weight0 = try file.entry("act_head.0.weight")
        XCTAssertEqual(weight0.dtype, "F16")
        XCTAssertEqual(weight0.shape, [256, 772])

        let bias0 = try file.entry("act_head.0.bias")
        XCTAssertEqual(bias0.dtype, "F16")
        XCTAssertEqual(bias0.shape, [256])

        let weight2 = try file.entry("act_head.2.weight")
        XCTAssertEqual(weight2.dtype, "F16")
        XCTAssertEqual(weight2.shape, [2, 256])

        let bias2 = try file.entry("act_head.2.bias")
        XCTAssertEqual(bias2.dtype, "F16")
        XCTAssertEqual(bias2.shape, [2])
    }

    private func row(_ collated: OrderedJSON, _ name: String) -> [Int32] {
        collated[name]!.arrayValue![0].arrayValue!.map { Int32($0.intValue!) }
    }
}

final class ErfGELUTests: XCTestCase {
    func testKnownValues() {
        XCTAssertEqual(ANEBackend.erfGELU(0), 0, accuracy: 1e-6)
        XCTAssertEqual(ANEBackend.erfGELU(1), 0.8413447, accuracy: 1e-6)
        XCTAssertEqual(ANEBackend.erfGELU(-1), -0.1586553, accuracy: 1e-6)
    }
}

final class CalibratorTests: XCTestCase {
    func testBucketKeysMatchPythonReference() {
        let expectedBuckets: [(k: Int, bucket: String)] = [
            (2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-10"), (10, "6-10"), (11, "11+"), (20, "11+")
        ]
        for qtype: QuestionType in [.choice, .score, .noul] {
            for (k, bucket) in expectedBuckets {
                let key = "\(qtype.name):\(bucket)"
                let calibrator = Calibrator(temperature: [1.0, 1.0, 1.0], temperatureByOptions: [key: 7.0])
                XCTAssertEqual(
                    calibrator.scale(qtype: qtype, optionCount: k), 7.0, "\(qtype.name) k=\(k) bucket=\(bucket)"
                )
            }
        }
    }

    func testFallsBackToBaseTemperatureWhenNoBucketOverride() {
        let calibrator = Calibrator(temperature: [1.0, 2.0, 3.0], temperatureByOptions: [:])
        XCTAssertEqual(calibrator.scale(qtype: .choice, optionCount: 4), 1.0)
        XCTAssertEqual(calibrator.scale(qtype: .score, optionCount: 4), 2.0)
        XCTAssertEqual(calibrator.scale(qtype: .noul, optionCount: 4), 3.0)
    }

    func testTemperatureByOptionsOverridesSoftmax() throws {
        let postProcessor = PostProcessor(
            calibrator: Calibrator(temperature: [1.0, 1.0, 1.0], temperatureByOptions: ["choice:2": 2.0])
        )
        let question = try NormalizedQuestion(.choice(instructions: "pick", options: ["a", "b"]))
        let answer = try postProcessor.answer(
            question: question, logits: [2.0, 0.0], actionLogits: [0, 0], optionCount: 2
        )
        let scaled: [Double] = [2.0 / 2.0, 0.0 / 2.0]
        let peak = scaled.max()!
        let exponentials = scaled.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        let expected = exponentials.map { $0 / total }
        XCTAssertEqual(answer.probabilities[0], expected[0], accuracy: 1e-9)
        XCTAssertEqual(answer.probabilities[1], expected[1], accuracy: 1e-9)
    }
}
