import Foundation

struct Calibrator {
    let temperature: [Double]
    let temperatureByOptions: [String: Double]

    func scale(qtype: QuestionType, optionCount: Int) -> Double {
        let size = optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
        let bucket = "\(qtype.name):\(size)"
        return temperatureByOptions[bucket] ?? temperature[Int(qtype.rawValue)]
    }
}

struct PostProcessor {
    let calibrator: Calibrator

    func answer(
        question: NormalizedQuestion,
        logits: [Float],
        actionLogits: [Float],
        optionCount k: Int
    ) throws -> LayaAnswer {
        guard actionLogits.allSatisfy({ $0.isFinite }), logits.allSatisfy({ $0.isFinite }) else {
            throw LayaError.invalidModelOutput("Non-finite Core ML outputs")
        }
        let act = softmax(actionLogits.map(Double.init))
        let scale = max(1e-3, calibrator.scale(qtype: question.type, optionCount: k))
        let probabilities = softmax(logits.prefix(k).map { Double($0) / scale })
        let labels = question.answerLabels
        var confidence = normalizedConfidence(probabilities, k)
        var choice: String?
        var score: Double?
        var noul: Double?

        switch question.type {
        case .choice:
            choice = labels[probabilities.firstIndex(of: probabilities.max()!)!]
        case .score:
            score = probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
        case .noul:
            noul = probabilities[1]
            confidence = max(probabilities[1], 1.0 - probabilities[1])
        }

        return LayaAnswer(
            probabilities: probabilities,
            labels: labels,
            choice: choice,
            score: score,
            noul: noul,
            confidence: confidence,
            actProbability: act[0]
        )
    }

    private func softmax(_ values: [Double]) -> [Double] {
        let peak = values.max()!
        let exponentials = values.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    private func normalizedConfidence(_ probabilities: [Double], _ k: Int) -> Double {
        guard k >= 2 else { return 1.0 }
        let entropy = -probabilities.reduce(0.0) { $0 + $1 * log(min(max($1, 1e-12), 1.0)) }
        return min(max(1.0 - entropy / log(Double(k)), 0.0), 1.0)
    }
}
