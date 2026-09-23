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
            // The directory is called Data, not Resources. `.copy` preserves the name inside the
            // generated SpektraFilm_SpektraFilm.bundle, and a directory called Resources sitting
            // next to that bundle's Info.plist makes codesign read it as a macOS-style bundle and
            // reject it: "bundle format unrecognized, invalid, or unsuitable". CI passes
            // CODE_SIGNING_ALLOWED=NO, so only a signed build sees it.
            resources: [.copy("Data")],
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
