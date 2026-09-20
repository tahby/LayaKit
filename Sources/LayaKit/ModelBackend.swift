protocol ModelBackend {
    func forward(_ batch: Batch) throws -> (logits: [Float], actionLogits: [Float])
}
