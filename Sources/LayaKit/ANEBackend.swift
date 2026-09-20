import CoreML
import Foundation

private struct EncoderConfig: Decodable {
    let hidden_size: Int
    let vocab_size: Int
    let local_attention: Int
}

final class ANEBackend: ModelBackend {
    private let model: MLModel
    private let weights: SafetensorsFile
    private let embeddingTable: UnsafeBufferPointer<Float16>
    private let typeEmbedding: [Float16]
    private let actionWeight0: [Float]
    private let actionBias0: [Float]
    private let actionWeight2: [Float]
    private let actionBias2: [Float]
    private let hiddenSize: Int
    private let vocabSize: Int
    private let windowRadius: Int
    private let length: Int
    private let maxOptions: Int

    init(bundle: URL, shape: BundleShape, padId: Int) throws {
        guard shape.batch_size == 1, shape.max_options == 32, !shape.flexible else {
            throw LayaError.unsupportedBundle("ANE runtime requires a fixed B1/K32 bundle")
        }

        let package = bundle.appendingPathComponent("model.mlpackage")
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw LayaError.missingFile(package.path)
        }
        let compiled = try ANEBackend.compiled(package: package, bundle: bundle)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: compiled, configuration: configuration)

        let encoderConfigURL = bundle.appendingPathComponent("encoder/config.json")
        guard let encoderConfigData = try? Data(contentsOf: encoderConfigURL) else {
            throw LayaError.missingFile(encoderConfigURL.path)
        }
        let encoderConfig = try JSONDecoder().decode(EncoderConfig.self, from: encoderConfigData)
        hiddenSize = encoderConfig.hidden_size
        vocabSize = encoderConfig.vocab_size
        windowRadius = encoderConfig.local_attention / 2
        length = shape.max_length
        maxOptions = shape.max_options

        try ANEBackend.validateInputSignature(model, hiddenSize: hiddenSize, length: length, maxOptions: maxOptions)

        let weightsURL = bundle.appendingPathComponent("host_weights.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LayaError.missingFile(weightsURL.path)
        }
        weights = try SafetensorsFile(url: weightsURL)

        try ANEBackend.validate(weights, name: "encoder.embeddings.tok_embeddings.weight", shape: [encoderConfig.vocab_size, hiddenSize])
        try ANEBackend.validate(weights, name: "type_emb.weight", shape: [3, hiddenSize])
        try ANEBackend.validate(weights, name: "act_head.0.weight", shape: [256, hiddenSize + 4])
        try ANEBackend.validate(weights, name: "act_head.0.bias", shape: [256])
        try ANEBackend.validate(weights, name: "act_head.2.weight", shape: [2, 256])
        try ANEBackend.validate(weights, name: "act_head.2.bias", shape: [2])

        embeddingTable = try weights.float16Pointer("encoder.embeddings.tok_embeddings.weight")
        typeEmbedding = try weights.float16Array("type_emb.weight")
        actionWeight0 = try weights.float32Array("act_head.0.weight")
        actionBias0 = try weights.float32Array("act_head.0.bias")
        actionWeight2 = try weights.float32Array("act_head.2.weight")
        actionBias2 = try weights.float32Array("act_head.2.bias")

        var warmup = [Int32](repeating: Int32(padId), count: length)
        warmup[0] = 1
        var mask = [Int32](repeating: 0, count: length)
        mask[0] = 1
        _ = try forward(
            Batch(
                inputIds: warmup,
                attentionMask: mask,
                markerPos: [Int32](repeating: 0, count: maxOptions),
                markerMask: [Int32](repeating: 0, count: maxOptions),
                qtype: [0],
                length: length,
                optionCount: 0
            )
        )
    }

    private static func validate(_ weights: SafetensorsFile, name: String, shape: [Int]) throws {
        let entry = try weights.entry(name)
        guard entry.shape == shape else {
            throw LayaError.unsupportedBundle("\(name) shape \(entry.shape) does not match expected \(shape)")
        }
    }

    private static func validateInputSignature(_ model: MLModel, hiddenSize: Int, length: Int, maxOptions: Int) throws {
        let expected: [String: [Int]] = [
            "embeddings": [1, hiddenSize, 1, length],
            "full_mask": [1, length, 1, length],
            "local_mask": [1, length, 1, length],
            "type_vectors": [1, hiddenSize, 1, 1],
            "marker_map": [1, length, 1, maxOptions]
        ]
        for (name, shape) in expected {
            guard let actual = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint?.shape.map(\.intValue),
                  actual == shape
            else {
                throw LayaError.unsupportedBundle("Package signature mismatch for \(name)")
            }
        }
    }

    private static func compiled(package: URL, bundle: URL) throws -> URL {
        let destination = bundle.appendingPathComponent("model.mlmodelc")
        let attributes = try? FileManager.default.attributesOfItem(atPath: destination.path)
        guard let packageDate = try FileManager.default.attributesOfItem(atPath: package.path)[.modificationDate] as? Date else {
            throw LayaError.unsupportedBundle(package.path)
        }
        if let compiledDate = attributes?[.modificationDate] as? Date, compiledDate >= packageDate {
            return destination
        }
        let temporary = try MLModel.compileModel(at: package)
        try? FileManager.default.removeItem(at: destination)
        do {
            try FileManager.default.moveItem(at: temporary, to: destination)
            return destination
        } catch {
            return temporary
        }
    }

    func forward(_ batch: Batch) throws -> (logits: [Float], actionLogits: [Float]) {
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "embeddings": embeddingsArray(batch),
            "full_mask": maskArray(batch, local: false),
            "local_mask": maskArray(batch, local: true),
            "type_vectors": typeVectorArray(batch),
            "marker_map": markerMapArray(batch)
        ])
        let outputs = try model.prediction(from: features)
        var logitsArray: MLMultiArray?
        var pooledArray: MLMultiArray?
        for name in outputs.featureNames {
            guard let value = outputs.featureValue(for: name)?.multiArrayValue else { continue }
            if value.shape[1] == 1 {
                logitsArray = value
            } else {
                pooledArray = value
            }
        }
        guard let logitsArray, let pooledArray else {
            throw LayaError.unsupportedBundle("Model output is missing logits or pooled state")
        }

        var logits = floats(logitsArray)
        let pooled = floats(pooledArray)
        for index in 0..<maxOptions {
            if batch.markerMask[index] == 0 {
                logits[index] = -1e4
            }
        }
        let peak = logits.max()!
        var probabilities = logits.map { expf($0 - peak) }
        let sum = probabilities.reduce(0, +)
        probabilities = probabilities.map { $0 / sum }
        let validCount = Float(batch.markerMask.reduce(0) { $0 + $1 })
        let k = max(validCount, 2)
        var entropy: Float = 0
        for p in probabilities {
            entropy += -(p * logf(max(p, 1e-9)))
        }
        entropy /= logf(k)
        let sorted = probabilities.sorted()
        let top1 = sorted[sorted.count - 1]
        let top2 = sorted[sorted.count - 2]
        let actionFeatures: [Float] = [top1, top1 - top2, entropy, k / 255.0]
        let actionInput = pooled + actionFeatures

        var hidden = [Float](repeating: 0, count: 256)
        for o in 0..<256 {
            var value: Float = actionBias0[o]
            let base = o * actionInput.count
            for i in 0..<actionInput.count {
                value += actionInput[i] * actionWeight0[base + i]
            }
            hidden[o] = ANEBackend.erfGELU(value)
        }
        var action = [Float](repeating: 0, count: 2)
        for o in 0..<2 {
            var value: Float = actionBias2[o]
            let base = o * 256
            for i in 0..<256 {
                value += hidden[i] * actionWeight2[base + i]
            }
            action[o] = value
        }
        return (logits, action)
    }

    static func erfGELU(_ x: Float) -> Float {
        let sqrt2 = 1.4142135623730951
        return Float(Double(x) * (1 + erf(Double(x) / sqrt2)) / 2)
    }

    private func embeddingsArray(_ batch: Batch) throws -> MLMultiArray {
        guard let result = try? MLMultiArray(shape: [1, NSNumber(value: hiddenSize), 1, NSNumber(value: length)], dataType: .float16) else {
            throw LayaError.unsupportedBundle("Unable to allocate embeddings input")
        }
        for pos in 0..<length {
            let id = Int(batch.inputIds[pos])
            guard id >= 0, id < vocabSize else {
                throw LayaError.unsupportedBundle("Token id \(id) outside checkpoint vocabulary")
            }
        }
        result.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            for pos in 0..<length {
                let rowBase = Int(batch.inputIds[pos]) * hiddenSize
                for d in 0..<hiddenSize {
                    buffer[d * length + pos] = embeddingTable[rowBase + d]
                }
            }
        }
        return result
    }

    private func maskArray(_ batch: Batch, local: Bool) throws -> MLMultiArray {
        guard let result = try? MLMultiArray(shape: [1, NSNumber(value: length), 1, NSNumber(value: length)], dataType: .float16) else {
            throw LayaError.unsupportedBundle("Unable to allocate mask input")
        }
        result.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            for j in 0..<length {
                let keyValid = batch.attentionMask[j] != 0
                for i in 0..<length {
                    let allow: Bool
                    if local {
                        let queryValid = batch.attentionMask[i] != 0
                        let windowOk = abs(i - j) <= windowRadius
                        allow = (windowOk || !queryValid) && keyValid
                    } else {
                        allow = keyValid
                    }
                    buffer[j * length + i] = allow ? 0 : -10000
                }
            }
        }
        return result
    }

    private func typeVectorArray(_ batch: Batch) throws -> MLMultiArray {
        guard let result = try? MLMultiArray(shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16) else {
            throw LayaError.unsupportedBundle("Unable to allocate type_vectors input")
        }
        let base = Int(batch.qtype[0]) * hiddenSize
        result.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            for d in 0..<hiddenSize {
                buffer[d] = typeEmbedding[base + d]
            }
        }
        return result
    }

    private func markerMapArray(_ batch: Batch) throws -> MLMultiArray {
        guard let result = try? MLMultiArray(shape: [1, NSNumber(value: length), 1, NSNumber(value: maxOptions)], dataType: .float16) else {
            throw LayaError.unsupportedBundle("Unable to allocate marker_map input")
        }
        result.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            for index in 0..<buffer.count {
                buffer[index] = 0
            }
            for slot in 0..<maxOptions {
                let pos = Int(batch.markerPos[slot])
                buffer[pos * maxOptions + slot] = 1
            }
        }
        return result
    }

    private func floats(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let a = shape[1]
        let b = shape[3]
        switch array.dataType {
        case .float16:
            return array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
                var values = [Float](repeating: 0, count: a * b)
                for i in 0..<a {
                    for j in 0..<b {
                        values[i * b + j] = Float(buffer[i * strides[1] + j * strides[3]])
                    }
                }
                return values
            }
        case .float32:
            return array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                var values = [Float](repeating: 0, count: a * b)
                for i in 0..<a {
                    for j in 0..<b {
                        values[i * b + j] = buffer[i * strides[1] + j * strides[3]]
                    }
                }
                return values
            }
        default:
            return (0..<array.count).map { array[$0].floatValue }
        }
    }
}
