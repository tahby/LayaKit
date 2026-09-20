public enum LayaError: Error {
    case tooManyTokens(count: Int, limit: Int)
    case tooManyOptions(count: Int, limit: Int)
    case invalidQuestion(String)
    case unsupportedBundle(String)
    case missingFile(String)
}
