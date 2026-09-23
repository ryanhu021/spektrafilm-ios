import Testing

@testable import SpektraFilm

/// The whole pipeline, and every stage boundary along the way.
///
/// The per-subsystem suites prove each operator is right on its own. These prove the operators are
/// wired in the right order with the right constants, which is a different failure mode and the one
/// a caller actually sees.
///
/// Every case runs in `lut_mode`, so the render is a deterministic per-pixel transform. Grain, glare
/// and the spatial operators have their own gates; mixing them in here would make these fixtures
/// depend on the RNG.
@Suite("End to end")
struct EndToEndTests {

    static let taps: [(tap: Tap, slug: String)] = [
        (.rgbPre, "rgb_pre"),
        (.logExposureFilm, "log_e_film"),
        (.cmyFilm, "cmy_film"),
        (.logExposurePrint, "log_e_print"),
        (.cmyPrint, "cmy_print"),
        (.rgbOut, "rgb_out"),
    ]

    static let cases: [(label: String, film: String, paper: String, output: String)] = [
        ("portra400_endura_srgb", "kodak_portra_400", "kodak_portra_endura", "sRGB"),
        ("velvia_endura_srgb", "fujifilm_velvia_100", "kodak_portra_endura", "sRGB"),
        ("vision3_500t_2383_srgb", "kodak_vision3_500t", "kodak_2383", "sRGB"),
        ("portra400_endura_p3", "kodak_portra_400", "kodak_portra_endura", "Display P3"),
    ]

    /// 18% grey through Portra 400 onto Portra Endura, in sRGB.
    ///
    /// The single number that says the pipeline is assembled correctly, and the one quoted in the
    /// README.
    @Test("18% grey matches the reference render")
    func midgreyAnchor() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let input = ImageBuffer(height: 4, width: 4, channels: 3, repeating: 0.184)
        let out = try Simulator(params).process(input)

        #expect(out.height == 4 && out.width == 4 && out.channels == 3)
        let expected = [0.4607145, 0.46055124, 0.46038181]
        for c in 0..<3 {
            let got = out[0, 0, c]
            #expect(
                abs(got - expected[c]) <= 1e-4,
                "channel \(c): \(got), reference \(expected[c]), delta \(got - expected[c])")
        }
    }

    @Test("every tap matches, across film, paper and output space", arguments: cases, taps)
    func tapParity(
        testCase: (label: String, film: String, paper: String, output: String),
        tap: (tap: Tap, slug: String)
    ) throws {
        var params = try RuntimePhotoParams.make(film: testCase.film, print: testCase.paper)
        params.camera.autoExposure = false
        params.debug.lutMode = true
        params.io.outputColourSpace = testCase.output

        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let out = try Simulator(params).process(input, inject: nil, collect: tap.tap)
        try expectParity(out.values, matches: "pipeline_\(testCase.label)_\(tap.slug)")
    }

    /// Scanning the negative directly skips the print entirely, which is a different topology.
    @Test(
        "scanning the negative takes the short topology",
        arguments: [
            (Tap.rgbPre, "rgb_pre"), (.logExposureFilm, "log_e_film"),
            (.cmyFilm, "cmy_film"), (.rgbOut, "rgb_out"),
        ]
    )
    func scanFilmParity(tap: Tap, slug: String) throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true
        params.io.scanFilm = true

        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let out = try Simulator(params).process(input, inject: nil, collect: tap)
        try expectParity(out.values, matches: "pipeline_scanfilm_portra400_\(slug)")
    }

    /// Asking for the print taps while scanning the negative has no path, and the topology should
    /// say so instead of returning something plausible.
    @Test("an unreachable tap is an error")
    func unreachableTap() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.debug.lutMode = true
        params.io.scanFilm = true
        let simulator = try Simulator(params)
        let input = ImageBuffer(height: 2, width: 2, channels: 3, repeating: 0.184)

        #expect(throws: (any Error).self) {
            try simulator.process(input, inject: nil, collect: .cmyPrint)
        }
    }

    /// A simulator is reusable: construction does the expensive per-film work, and running it twice
    /// must give the same answer.
    @Test("a simulator is reusable across frames")
    func reusable() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let simulator = try Simulator(params)
        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let first = try simulator.process(input)
        let second = try simulator.process(input)
        #expect(first.values == second.values)
    }

    @Test("timings are recorded for every node that fired")
    func timings() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let simulator = try Simulator(params)
        _ = try simulator.process(ImageBuffer(height: 4, width: 4, channels: 3, repeating: 0.184))

        #expect(simulator.timings.count == 6, "\(simulator.timings.keys.sorted())")
        #expect(simulator.elapsed != nil)
        #expect(simulator.timings.values.allSatisfy { $0 >= 0 })
    }
}
