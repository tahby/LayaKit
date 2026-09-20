import Foundation

final class SafetensorsFile {
    struct Entry {
        let dtype: String
        let shape: [Int]
        let byteRange: Range<Int>
    }

    private let storage: NSData
    private let baseOffset: Int
    private let entries: [String: Entry]

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else {
            throw LayaError.unsupportedBundle(url.path)
        }
        let rawHeaderLength = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }
        guard let headerLength = Int(exactly: rawHeaderLength), headerLength <= data.count - 8 else {
            throw LayaError.unsupportedBundle(url.path)
        }
        let headerData = data.subdata(in: 8..<(8 + headerLength))
        guard let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw LayaError.unsupportedBundle(url.path)
        }
        let base = 8 + headerLength
        var parsed: [String: Entry] = [:]
        for (key, value) in object {
            guard key != "__metadata__" else { continue }
            guard let fields = value as? [String: Any],
                  let dtype = fields["dtype"] as? String,
                  let shape = fields["shape"] as? [Int],
                  let offsets = fields["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw LayaError.unsupportedBundle(url.path)
            }
            guard offsets[0] >= 0, offsets[0] <= offsets[1], base + offsets[1] <= data.count else {
                throw LayaError.unsupportedBundle(url.path)
            }
            guard let elementSize = SafetensorsFile.byteWidth(of: dtype),
                  offsets[1] - offsets[0] == shape.reduce(1, *) * elementSize
            else {
                throw LayaError.unsupportedBundle(url.path)
            }
            parsed[key] = Entry(dtype: dtype, shape: shape, byteRange: offsets[0]..<offsets[1])
        }
        entries = parsed
        baseOffset = base
        storage = data as NSData
    }

    private static func byteWidth(of dtype: String) -> Int? {
        switch dtype {
        case "F16", "BF16": return 2
        case "F32", "I32", "U32": return 4
        case "F64", "I64", "U64": return 8
        case "I8", "U8", "BOOL": return 1
        default: return nil
        }
    }

    func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else {
            throw LayaError.unsupportedBundle("Missing tensor \(name)")
        }
        return entry
    }

    func float16Pointer(_ name: String) throws -> UnsafeBufferPointer<Float16> {
        let entry = try entry(name)
        guard entry.dtype == "F16" else {
            throw LayaError.unsupportedBundle("\(name) is not float16")
        }
        let base = storage.bytes.advanced(by: baseOffset + entry.byteRange.lowerBound)
        let count = entry.byteRange.count / MemoryLayout<Float16>.size
        return UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float16.self), count: count)
    }

    func float16Array(_ name: String) throws -> [Float16] {
        Array(try float16Pointer(name))
    }

    func float32Array(_ name: String) throws -> [Float] {
        try float16Pointer(name).map(Float.init)
    }
}
