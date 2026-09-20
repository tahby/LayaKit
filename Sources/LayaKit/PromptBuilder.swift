struct TokenizedItem {
    let ids: [Int]
    let markers: [Int]
    let qtype: QuestionType
}

struct PromptBuilder {
    let tokenizer: LayaTokenizer
    let maxLen: Int
    let headMaxLen: Int

    private func prefix(_ question: NormalizedQuestion) -> (ids: [Int], markers: [Int]) {
        let special = tokenizer.special
        let options = question.optionTexts
        let instructions = question.instructions.replacingOccurrences(of: special.maskText, with: " ")
        var headIds = tokenizer.encode("\(question.type.name) question: \(instructions)")
        var optionIds = options.map { option in
            [special.mask]
                + tokenizer.encode(" " + option.replacingOccurrences(of: special.maskText, with: " ")).prefix(48)
        }
        var optionBudget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        if optionBudget < 16 {
            let per = max(4, (headMaxLen - 16) / max(1, optionIds.count))
            optionIds = optionIds.map { Array($0.prefix(per)) }
            optionBudget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        }
        headIds = Array(headIds.prefix(max(8, optionBudget)))

        var ids = [special.cls] + headIds + [special.sep]
        var markers: [Int] = []
        for option in optionIds {
            markers.append(ids.count)
            ids.append(contentsOf: option)
        }
        ids.append(special.sep)
        return (ids, markers)
    }

    func build(state: String, question: NormalizedQuestion) throws -> TokenizedItem {
        let (prefixIds, markers) = prefix(question)
        let room = max(0, maxLen - prefixIds.count - 1)
        let stateIds = tokenizer.encode(state.replacingOccurrences(of: tokenizer.special.maskText, with: " "))
        let ids = prefixIds + stateIds.prefix(room) + [tokenizer.special.sep]
        let kept = markers.filter { $0 < maxLen }
        guard kept.count == question.optionTexts.count else {
            throw LayaError.invalidQuestion("Question has too many options for the token budget")
        }
        return TokenizedItem(ids: Array(ids.prefix(maxLen)), markers: kept, qtype: question.type)
    }
}
