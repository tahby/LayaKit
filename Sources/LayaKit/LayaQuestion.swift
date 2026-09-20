import Foundation

public enum LayaQuestion {
    case choice(instructions: String, options: [String], descriptions: [String: String]? = nil)
    case score(instructions: String, levels: [String])
    case noul(instructions: String, falseText: String? = nil, trueText: String? = nil)
}

enum QuestionType: Int32 {
    case choice = 0
    case score = 1
    case noul = 2

    var name: String {
        switch self {
        case .choice: return "choice"
        case .score: return "score"
        case .noul: return "noul"
        }
    }
}

struct NormalizedQuestion {
    enum Criteria {
        case choice(labels: [String], descriptions: [String?])
        case score(levels: [String])
        case noul(falseText: String?, trueText: String?)
    }

    let instructions: String
    let criteria: Criteria

    init(_ question: LayaQuestion) throws {
        switch question {
        case let .choice(instructions, options, descriptions):
            guard !options.isEmpty else {
                throw LayaError.invalidQuestion("Choice criteria must be a nonempty dictionary or list")
            }
            guard Set(options).count == options.count else {
                throw LayaError.invalidQuestion("Choice labels must be unique")
            }
            self.instructions = instructions
            self.criteria = .choice(labels: options, descriptions: options.map { descriptions?[$0] })
        case let .score(instructions, levels):
            guard !levels.isEmpty else {
                throw LayaError.invalidQuestion("Score criteria must be a nonempty list")
            }
            self.instructions = instructions
            self.criteria = .score(levels: levels)
        case let .noul(instructions, falseText, trueText):
            self.instructions = instructions
            self.criteria = .noul(falseText: falseText, trueText: trueText)
        }
    }

    var type: QuestionType {
        switch criteria {
        case .choice: return .choice
        case .score: return .score
        case .noul: return .noul
        }
    }

    var optionTexts: [String] {
        switch criteria {
        case let .choice(labels, descriptions):
            return zip(labels, descriptions).map { label, description in
                guard let description, !description.isEmpty else { return label }
                return "\(label): \(description)"
            }
        case let .score(levels):
            return levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case let .noul(falseText, trueText):
            let no = (falseText?.isEmpty == false) ? falseText! : "no, the statement does not hold"
            let yes = (trueText?.isEmpty == false) ? trueText! : "yes, the statement holds"
            return ["false: \(no)", "true: \(yes)"]
        }
    }

    var answerLabels: [String] {
        switch criteria {
        case let .choice(labels, _): return labels
        case let .score(levels): return levels
        case .noul: return ["false", "true"]
        }
    }
}
