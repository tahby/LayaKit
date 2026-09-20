import Foundation

indirect enum OrderedJSON {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([OrderedJSON])
    case object([(key: String, value: OrderedJSON)])

    subscript(key: String) -> OrderedJSON? {
        guard case let .object(entries) = self else { return nil }
        return entries.first { $0.key == key }?.value
    }

    var stringValue: String? { if case let .string(text) = self { return text } else { return nil } }
    var boolValue: Bool? { if case let .bool(flag) = self { return flag } else { return nil } }
    var doubleValue: Double? { if case let .number(value) = self { return value } else { return nil } }
    var intValue: Int? { doubleValue.map(Int.init) }
    var arrayValue: [OrderedJSON]? { if case let .array(items) = self { return items } else { return nil } }
    var entries: [(key: String, value: OrderedJSON)]? {
        if case let .object(entries) = self { return entries } else { return nil }
    }
    var isNull: Bool { if case .null = self { return true } else { return false } }

    static func parse(_ data: Data) throws -> OrderedJSON {
        var parser = Parser(bytes: Array(data))
        return try parser.value()
    }
}

enum OrderedJSONError: Error, CustomStringConvertible {
    case malformed(String)

    var description: String {
        switch self {
        case let .malformed(reason): return "Malformed JSON: \(reason)"
        }
    }
}

private let maxNestingDepth = 512

private struct Parser {
    let bytes: [UInt8]
    var index = 0
    var depth = 0

    mutating func byte() throws -> UInt8 {
        guard index < bytes.count else { throw OrderedJSONError.malformed("unexpected end of input") }
        return bytes[index]
    }

    mutating func value() throws -> OrderedJSON {
        skipWhitespace()
        switch try byte() {
        case UInt8(ascii: "{"): return try object()
        case UInt8(ascii: "["): return try array()
        case UInt8(ascii: "\""): return .string(try string())
        case UInt8(ascii: "t"): index += 4; return .bool(true)
        case UInt8(ascii: "f"): index += 5; return .bool(false)
        case UInt8(ascii: "n"): index += 4; return .null
        default: return .number(try number())
        }
    }

    mutating func skipWhitespace() {
        while index < bytes.count, bytes[index] == 0x20 || bytes[index] == 0x0A || bytes[index] == 0x0D || bytes[index] == 0x09 {
            index += 1
        }
    }

    mutating func object() throws -> OrderedJSON {
        depth += 1
        guard depth <= maxNestingDepth else { throw OrderedJSONError.malformed("nesting too deep") }
        defer { depth -= 1 }
        index += 1
        var entries: [(key: String, value: OrderedJSON)] = []
        skipWhitespace()
        if try byte() == UInt8(ascii: "}") { index += 1; return .object(entries) }
        while true {
            skipWhitespace()
            let key = try string()
            skipWhitespace()
            index += 1
            entries.append((key, try value()))
            skipWhitespace()
            let separator = try byte()
            index += 1
            if separator == UInt8(ascii: "}") { return .object(entries) }
        }
    }

    mutating func array() throws -> OrderedJSON {
        depth += 1
        guard depth <= maxNestingDepth else { throw OrderedJSONError.malformed("nesting too deep") }
        defer { depth -= 1 }
        index += 1
        var items: [OrderedJSON] = []
        skipWhitespace()
        if try byte() == UInt8(ascii: "]") { index += 1; return .array(items) }
        while true {
            items.append(try value())
            skipWhitespace()
            let separator = try byte()
            index += 1
            if separator == UInt8(ascii: "]") { return .array(items) }
        }
    }

    mutating func string() throws -> String {
        index += 1
        var raw: [UInt8] = []
        while try byte() != UInt8(ascii: "\"") {
            if try byte() == UInt8(ascii: "\\") {
                index += 1
                switch try byte() {
                case UInt8(ascii: "n"): raw.append(0x0A)
                case UInt8(ascii: "t"): raw.append(0x09)
                case UInt8(ascii: "r"): raw.append(0x0D)
                case UInt8(ascii: "b"): raw.append(0x08)
                case UInt8(ascii: "f"): raw.append(0x0C)
                case UInt8(ascii: "u"):
                    guard index + 4 < bytes.count else { throw OrderedJSONError.malformed("truncated unicode escape") }
                    let hex = String(decoding: bytes[(index + 1)...(index + 4)], as: UTF8.self)
                    guard let code = UInt32(hex, radix: 16), let scalar = UnicodeScalar(code) else {
                        throw OrderedJSONError.malformed("invalid unicode escape")
                    }
                    raw.append(contentsOf: Array(String(scalar).utf8))
                    index += 4
                default: raw.append(try byte())
                }
            } else {
                raw.append(try byte())
            }
            index += 1
        }
        index += 1
        return String(decoding: raw, as: UTF8.self)
    }

    mutating func number() throws -> Double {
        let start = index
        while index < bytes.count, bytes[index] != UInt8(ascii: ","), bytes[index] != UInt8(ascii: "]"),
              bytes[index] != UInt8(ascii: "}"), bytes[index] > 0x20 {
            index += 1
        }
        guard let result = Double(String(decoding: bytes[start..<index], as: UTF8.self)) else {
            throw OrderedJSONError.malformed("invalid number")
        }
        return result
    }
}
