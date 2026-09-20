// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "LayaKit", targets: ["LayaKit"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4")
    ],
    targets: [
        .target(
            name: "LayaKit",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .testTarget(
            name: "LayaKitTests",
            dependencies: ["LayaKit"],
            resources: [.copy("Fixtures/fixtures.json")]
        )
    ]
)
