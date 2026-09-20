import CoreML
import Foundation

final class GeneralBackend: ModelBackend {
    private let model: MLModel

    init(bundle: URL, shape: BundleShape, padId: Int) throws {
        let package = bundle.appendingPathComponent("model.mlpackage")
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw LayaError.missingFile(package.path)
        }
        let compiled = try GeneralBackend.compiled(package: package, bundle: bundle)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        model = try MLModel(contentsOf: compiled, configuration: configuration)

        var warmup = [Int32](repeating: Int32(padId), count: shape.min_length)
        warmup[0] = 1
        var mask = [Int32](repeating: 0, count: shape.min_length)
        mask[0] = 1
        _ = try forward(
            Batch(
                inputIds: warmup,
                attentionMask: mask,
                markerPos: [Int32](repeating: 0, count: shape.max_options),
                markerMask: [Int32](repeating: 0, count: shape.max_options),
                qtype: [0],
                length: shape.min_length,
                optionCount: 0
            )
        )
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
            "input_ids": array(batch.inputIds, shape: [1, batch.length]),
            "attention_mask": array(batch.attentionMask, shape: [1, batch.length]),
            "marker_pos": array(batch.markerPos, shape: [1, batch.markerPos.count]),
            "marker_mask": array(batch.markerMask, shape: [1, batch.markerMask.count]),
            "qtype": array(batch.qtype, shape: [1])
        ])
        let outputs = try model.prediction(from: features)
        guard let logits = outputs.featureValue(for: "logits")?.multiArrayValue else {
            throw LayaError.unsupportedBundle("Model output is missing logits")
        }
        guard let actionLogits = outputs.featureValue(for: "action_logits")?.multiArrayValue else {
            throw LayaError.unsupportedBundle("Model output is missing action_logits")
        }
        return (floats(logits), floats(actionLogits))
    }

    private func array(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        guard let result = try? MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32) else {
            throw LayaError.unsupportedBundle("Unable to allocate model input")
        }
        result.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            _ = buffer.update(fromContentsOf: values)
        }
        return result
    }

    private func floats(_ array: MLMultiArray) -> [Float] {
        array.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
    }
}
