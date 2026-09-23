import Testing

@testable import SpektraFilm

/// A real photograph through four film and paper combinations.
///
/// The synthetic ramps and patches in other suites test one operator at a time. This suite tests
/// what a user sees: skin, foliage, specular highlights and deep shadow in one frame. A midtone hue
/// shift or a crushed toe shows up here and in no other suite.
///
/// The negative tap is included because the app displays it.
@Suite("Photograph parity")
struct PhotographParityTests {

    static let combinations: [(label: String, film: String, paper: String)] = [
        ("portra400_endura", "kodak_portra_400", "kodak_portra_endura"),
        // A slide printed onto negative paper. The result is pale and low contrast, because a
        // positive's base density is far higher than the enlarger exposure assumes. Nothing clips,
        // and the negative tap holds a normal image, so this still exercises the positive-film
        // coupler path end to end.
        ("velvia_endura", "fujifilm_velvia_100", "kodak_portra_endura"),
        ("vision3_500t_2383", "kodak_vision3_500t", "kodak_2383"),
        ("gold200_supra", "kodak_gold_200", "kodak_supra_endura"),
    ]

    @Test("the print matches", arguments: combinations)
    func print(combination: (label: String, film: String, paper: String)) throws {
        var params = try RuntimePhotoParams.make(
            film: combination.film, print: combination.paper)
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let input = try Golden("photo_input").imageBuffer()
        let out = try Simulator(params).process(input)
        try expectParity(out.values, matches: "photo_\(combination.label)")
    }

    @Test("the negative matches", arguments: combinations)
    func negative(combination: (label: String, film: String, paper: String)) throws {
        var params = try RuntimePhotoParams.make(
            film: combination.film, print: combination.paper)
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let input = try Golden("photo_input").imageBuffer()
        let out = try Simulator(params).process(input, inject: nil, collect: .cmyFilm)
        try expectParity(out.values, matches: "photo_\(combination.label)_negative")
    }

    /// Different stocks must render differently. A wiring bug that ignored the profile would pass
    /// every per-operator test and every parity fixture that compares one stock to itself.
    ///
    /// The closest pair, Portra 400 and Gold 200, differs by 0.0725 at its worst pixel. The 0.02
    /// gate leaves margin below that and still fails two identical renders.
    @Test("the four combinations are distinguishable from each other")
    func stocksDiffer() throws {
        var renders: [String: [Double]] = [:]
        for combination in Self.combinations {
            renders[combination.label] = try Golden("photo_\(combination.label)").values
        }
        for a in Self.combinations.map(\.label) {
            for b in Self.combinations.map(\.label) where a < b {
                let report = parity(renders[a]!, renders[b]!)
                #expect(
                    report.maxAbsolute > 0.02,
                    "\(a) and \(b) differ by only \(report.maxAbsolute), which is suspiciously close")
            }
        }
    }

    /// A negative should read as a negative: dense where the scene was bright.
    @Test("the negative is inverted relative to the print")
    func negativeIsInverted() throws {
        let input = try Golden("photo_input").values
        let negative = try Golden("photo_portra400_endura_negative").values

        // Correlate scene luminance against negative density across the frame. Density rises with
        // exposure on a negative, so the correlation should be strongly positive.
        var meanScene = 0.0
        var meanDensity = 0.0
        let pixels = input.count / 3
        for p in 0..<pixels {
            meanScene += (input[p * 3] + input[p * 3 + 1] + input[p * 3 + 2]) / 3
            meanDensity += (negative[p * 3] + negative[p * 3 + 1] + negative[p * 3 + 2]) / 3
        }
        meanScene /= Double(pixels)
        meanDensity /= Double(pixels)

        var covariance = 0.0
        var sceneVariance = 0.0
        var densityVariance = 0.0
        for p in 0..<pixels {
            let s = (input[p * 3] + input[p * 3 + 1] + input[p * 3 + 2]) / 3 - meanScene
            let d = (negative[p * 3] + negative[p * 3 + 1] + negative[p * 3 + 2]) / 3 - meanDensity
            covariance += s * d
            sceneVariance += s * s
            densityVariance += d * d
        }
        let correlation = covariance / (sceneVariance * densityVariance).squareRoot()
        #expect(correlation > 0.5, "scene to negative density correlation is \(correlation)")
    }
}
