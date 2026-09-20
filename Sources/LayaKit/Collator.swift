struct BundleShape: Decodable {
    let batch_size: Int
    let max_length: Int
    let min_length: Int
    let max_options: Int
    let flexible: Bool
    let lengths: [Int]?
}

struct Batch {
    let inputIds: [Int32]
    let attentionMask: [Int32]
    let markerPos: [Int32]
    let markerMask: [Int32]
    let qtype: [Int32]
    let length: Int
    let optionCount: Int
}

struct Collator {
    let shape: BundleShape
    let padId: Int

    func collate(_ item: TokenizedItem) throws -> Batch {
        let tokenCount = item.ids.count
        guard tokenCount <= shape.max_length else {
            throw LayaError.tooManyTokens(count: tokenCount, limit: shape.max_length)
        }
        guard item.markers.count <= shape.max_options else {
            throw LayaError.tooManyOptions(count: item.markers.count, limit: shape.max_options)
        }
        let length: Int
        if let lengths = shape.lengths, !lengths.isEmpty {
            guard let matched = lengths.first(where: { $0 >= tokenCount }) else {
                throw LayaError.tooManyTokens(count: tokenCount, limit: shape.max_length)
            }
            length = matched
        } else if shape.flexible {
            length = min(shape.max_length, max(shape.min_length, ((tokenCount + 15) / 16) * 16))
        } else {
            length = shape.max_length
        }
        let options = shape.max_options
        var inputIds = [Int32](repeating: Int32(padId), count: length)
        var attentionMask = [Int32](repeating: 0, count: length)
        var markerPos = [Int32](repeating: 0, count: options)
        var markerMask = [Int32](repeating: 0, count: options)
        for (index, id) in item.ids.enumerated() {
            inputIds[index] = Int32(id)
            attentionMask[index] = 1
        }
        for (index, marker) in item.markers.enumerated() {
            markerPos[index] = Int32(marker)
            markerMask[index] = 1
        }
        return Batch(
            inputIds: inputIds,
            attentionMask: attentionMask,
            markerPos: markerPos,
            markerMask: markerMask,
            qtype: [item.qtype.rawValue],
            length: length,
            optionCount: item.markers.count
        )
    }
}
