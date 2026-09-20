import CoreML

func modelOutputFloats(_ array: MLMultiArray) -> [Float] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    switch array.dataType {
    case .float16:
        return array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
            flattenedFloats(shape: shape, strides: strides) { Float(buffer[$0]) }
        }
    case .float32:
        return array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            flattenedFloats(shape: shape, strides: strides) { buffer[$0] }
        }
    default:
        return (0..<array.count).map { array[$0].floatValue }
    }
}

private func flattenedFloats(shape: [Int], strides: [Int], read: (Int) -> Float) -> [Float] {
    let count = shape.reduce(1, *)
    var values = [Float](repeating: 0, count: count)
    var indices = [Int](repeating: 0, count: shape.count)
    for flatIndex in 0..<count {
        let offset = zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }
        values[flatIndex] = read(offset)
        var dim = shape.count - 1
        while dim >= 0 {
            indices[dim] += 1
            if indices[dim] < shape[dim] { break }
            indices[dim] = 0
            dim -= 1
        }
    }
    return values
}
