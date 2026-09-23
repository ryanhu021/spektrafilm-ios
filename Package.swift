// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SpektraFilm",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "SpektraFilm", targets: ["SpektraFilm"]),
        // Peak-memory profiler. Memory, not time, is what caps export size, so the instrument ships
        // with the code it measures.
        .executable(name: "memprofile", targets: ["memprofile"]),
    ],
    targets: [
        .target(
            name: "SpektraFilm",
            resources: [.copy("Resources")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "memprofile",
            dependencies: ["SpektraFilm"],
            path: "Tools/memprofile",
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
