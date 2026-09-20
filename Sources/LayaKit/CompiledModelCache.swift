import CoreML
import Foundation

enum CompiledModelCache {
    static func compiled(package: URL, bundle: URL) throws -> URL {
        let destination = bundle.appendingPathComponent("model.mlmodelc")
        guard let packageDate = newestModificationDate(at: package) else {
            throw LayaError.unsupportedBundle(package.path)
        }
        let attributes = try? FileManager.default.attributesOfItem(atPath: destination.path)
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

    private static func newestModificationDate(at url: URL) -> Date? {
        let manager = FileManager.default
        guard let attributes = try? manager.attributesOfItem(atPath: url.path),
              let selfDate = attributes[.modificationDate] as? Date
        else {
            return nil
        }
        guard attributes[.type] as? FileAttributeType == .typeDirectory,
              let enumerator = manager.enumerator(at: url, includingPropertiesForKeys: [.contentModificationDateKey])
        else {
            return selfDate
        }
        var newest = selfDate
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]),
                  let date = values.contentModificationDate
            else { continue }
            if date > newest { newest = date }
        }
        return newest
    }
}
