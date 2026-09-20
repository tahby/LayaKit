import Foundation
import Tokenizers

struct SpecialTokens {
    let cls: Int
    let sep: Int
    let pad: Int
    let mask: Int
    let maskText: String
}

final class LayaTokenizer {
    private let backend: any Tokenizer
    let special: SpecialTokens

    init(folder: URL) async throws {
        let configURL = folder.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            throw LayaError.missingFile(configURL.path)
        }
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path) else {
            throw LayaError.missingFile(folder.appendingPathComponent("tokenizer.json").path)
        }
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LayaError.unsupportedBundle("Tokenizer config is not a valid JSON object")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        backend = tokenizer

        func token(_ name: String) throws -> String {
            let raw = config[name]
            let text = (raw as? String) ?? ((raw as? [String: Any])?["content"] as? String)
            guard let text else { throw LayaError.unsupportedBundle("Tokenizer is missing a valid \(name)") }
            return text
        }
        func identifier(_ name: String) throws -> Int {
            let text = try token(name)
            guard let id = tokenizer.convertTokenToId(text) else {
                throw LayaError.unsupportedBundle("Tokenizer is missing a valid \(name)")
            }
            return id
        }
        special = SpecialTokens(
            cls: try identifier("cls_token"),
            sep: try identifier("sep_token"),
            pad: try identifier("pad_token"),
            mask: try identifier("mask_token"),
            maskText: try token("mask_token")
        )
    }

    func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        return backend.encode(text: text, addSpecialTokens: false)
    }
}
