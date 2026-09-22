// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SpektraFilm",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "SpektraFilm", targets: ["SpektraFilm"])
    ],
    targets: [
        .target(
            name: "SpektraFilm",
            resources: [.copy("Resources")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SpektraFilmTests",
            dependencies: ["SpektraFilm"],
            resources: [.copy("Goldens")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
