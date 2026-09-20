import Foundation

private struct BundleManifest: Decodable {
    let format: String
    let format_version: Int?
    let shape: BundleShape
}

private struct AgentConfig: Decodable {
    let max_len: Int?
    let head_max_len: Int?
    let temperature: [Double]?
    let temperature_by_options: [String: Double]?
}

public final class LayaAgent: @unchecked Sendable {
    let promptBuilder: PromptBuilder
    let collator: Collator
    let backend: ModelBackend
    let postProcessor: PostProcessor

    public init(bundle: URL) async throws {
        let manifestURL = bundle.appendingPathComponent("coreml_config.json")
        guard let manifestData = try? Data(contentsOf: manifestURL) else {
            throw LayaError.missingFile(manifestURL.path)
        }
        let manifest = try JSONDecoder().decode(BundleManifest.self, from: manifestData)
        guard manifest.format == "laya-coreml" || manifest.format == "laya-coreml-ane" else {
            throw LayaError.unsupportedBundle(manifest.format)
        }
        guard manifest.format_version == 1 else {
            throw LayaError.unsupportedBundle("format_version \(manifest.format_version.map(String.init) ?? "missing")")
        }

        let configURL = bundle.appendingPathComponent("rl_agent_config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw LayaError.missingFile(configURL.path)
        }
        let config = try JSONDecoder().decode(AgentConfig.self, from: configData)
        let temperature = config.temperature ?? [1.0, 1.0, 1.0]
        let temperatureByOptions = config.temperature_by_options ?? [:]
        let allTemperatures = temperature + Array(temperatureByOptions.values)
        guard temperature.count == 3, allTemperatures.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw LayaError.unsupportedBundle("Calibration temperatures must be finite and positive")
        }

        let tokenizer = try await LayaTokenizer(folder: bundle.appendingPathComponent("tokenizer"))
        promptBuilder = PromptBuilder(
            tokenizer: tokenizer,
            maxLen: config.max_len ?? 512,
            headMaxLen: config.head_max_len ?? 192
        )
        collator = Collator(shape: manifest.shape, padId: tokenizer.special.pad)
        postProcessor = PostProcessor(
            calibrator: Calibrator(
                temperature: temperature,
                temperatureByOptions: temperatureByOptions
            )
        )
        if manifest.format == "laya-coreml-ane" {
            backend = try ANEBackend(
                bundle: bundle,
                shape: manifest.shape,
                padId: tokenizer.special.pad
            )
        } else {
            backend = try GeneralBackend(
                bundle: bundle,
                shape: manifest.shape,
                padId: tokenizer.special.pad
            )
        }
    }

    public func predict(state: String, question: LayaQuestion) throws -> LayaAnswer {
        let normalized = try NormalizedQuestion(question)
        let item = try promptBuilder.build(state: state, question: normalized)
        let batch = try collator.collate(item)
        let (logits, actionLogits) = try backend.forward(batch)
        return try postProcessor.answer(
            question: normalized,
            logits: logits,
            actionLogits: actionLogits,
            optionCount: batch.optionCount
        )
    }
}
