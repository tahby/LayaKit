// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "LayaKit", targets: ["LayaKit"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0")
    ],
    targets: [
        .target(
            name: "LayaKit",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .executableTarget(
            name: "laya-cli",
            dependencies: [
                "LayaKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "LayaKitTests",
            dependencies: ["LayaKit"],
            resources: [.copy("Fixtures/fixtures.json")]
        )
    ]
)
