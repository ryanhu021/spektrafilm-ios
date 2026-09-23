import Testing

@testable import SpektraFilm

/// The print-curve morph, `utils/morph_curves.apply_print_curves_morph`.
///
/// The curves are pure arithmetic plus one Brent solve converged to 1e-10, so they are compared at
/// 1e-9, far inside the render tolerance. The synthetic models cover branches no shipped paper
/// reaches: positive profiles, the index collisions of a two-layer model, the unscaled layer of a
/// four-layer model, and the sigma floor.
@Suite("Print-curve morph parity")
struct MorphParityTests {

    static let cases: [(label: String, configure: @Sendable (inout PrintCurvesMorphParams) -> Void)] = [
        ("identity", { _ in }),
        ("contrast_up", { $0.gammaFactor = 1.3 }),
        (
            "split_speed",
            {
                $0.gammaFactor = 0.7
                $0.gammaFactorFast = 1.2
                $0.gammaFactorSlow = 0.9
            }
        ),
        (
            "rgb",
            {
                $0.gammaFactorRed = 1.1
                $0.gammaFactorGreen = 0.95
                $0.gammaFactorBlue = 1.05
            }
        ),
        ("exhaustion", { $0.developerExhaustion = 0.5 }),
        (
            "combined",
            {
                $0.gammaFactor = 1.2
                $0.gammaFactorFast = 0.8
                $0.developerExhaustion = 0.8
            }
        ),
    ]

    static func params(_ label: String) -> PrintCurvesMorphParams {
        var params = PrintCurvesMorphParams(active: true)
        if label == "floor" {
            params.gammaFactor = 6.0
        } else {
            cases.first { $0.label == label }!.configure(&params)
        }
        return params
    }

    @Test(
        "shipped papers match the oracle",
        arguments: ["kodak_portra_endura", "kodak_2383", "kodak_supra_endura"])
    func papers(paper: String) throws {
        let profile = try ProfileLibrary.load(paper)
        let axis = try Golden("morph_\(paper)_log_exposure").values
        #expect(axis == profile.data.logExposure)
        for (label, _) in Self.cases {
            let curves = try PrintCurvesMorph.apply(
                logExposure: axis, model: profile.data.densityCurvesModel,
                params: Self.params(label), positive: profile.isPositive)
            try expectParity(
                curves, matches: "morph_\(paper)_\(label)", maxAbsolute: 1e-9,
                rootMeanSquare: 1e-10)
        }
    }

    static let twoLayer = DensityCurvesModel(
        channelCount: 3, layerCount: 2,
        centers: [0.2, 1.1, 0.3, 1.0, 0.1, 1.3],
        amplitudes: [1.2, 0.9, 1.0, 1.1, 0.8, 1.3],
        sigmas: [0.4, 0.3, 0.35, 0.45, 0.5, 0.25])

    static let fourLayer = DensityCurvesModel(
        channelCount: 3, layerCount: 4,
        centers: [0.9, -0.2, 0.4, 1.5, 0.0, 0.6, 1.2, -0.5, 0.3, 0.3, 1.0, 1.8],
        amplitudes: [0.5, 0.6, 0.7, 0.4, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.7],
        sigmas: [0.3, 0.2, 0.25, 0.35, 0.2, 0.3, 0.4, 0.25, 0.3, 0.3, 0.2, 0.4])

    @Test("synthetic models match the oracle", arguments: ["two_layer", "four_layer"])
    func synthetic(name: String) throws {
        let model = name == "two_layer" ? Self.twoLayer : Self.fourLayer
        let axis = try Golden("morph_synthetic_log_exposure").values
        for positive in [false, true] {
            for label in Self.cases.map(\.label) + ["floor"] {
                let curves = try PrintCurvesMorph.apply(
                    logExposure: axis, model: model, params: Self.params(label),
                    positive: positive)
                let type = positive ? "positive" : "negative"
                try expectParity(
                    curves, matches: "morph_\(name)_\(type)_\(label)", maxAbsolute: 1e-9,
                    rootMeanSquare: 1e-10)
            }
        }
    }

    /// Exhaustion must move the curves, and the shift must restore the density at zero exposure.
    /// Without the shift the exhaustion case still matches its own golden loosely, but D(0) drifts.
    @Test("developer exhaustion preserves the density at zero log exposure")
    func exhaustionPreservesZero() throws {
        let profile = try ProfileLibrary.load("kodak_portra_endura")
        let model = profile.data.densityCurvesModel
        let plain = try PrintCurvesMorph.apply(
            logExposure: [0.0], model: model, params: PrintCurvesMorphParams(active: false),
            positive: false)
        let exhausted = try PrintCurvesMorph.apply(
            logExposure: [0.0, 1.0], model: model, params: Self.params("exhaustion"),
            positive: false)
        for c in 0..<3 {
            #expect(abs(exhausted[c] - plain[c]) < 1e-9)
        }
        let unexhausted = try PrintCurvesMorph.apply(
            logExposure: [1.0], model: model, params: PrintCurvesMorphParams(active: false),
            positive: false)
        #expect(abs(exhausted[3] - unexhausted[0]) > 1e-3)
    }

    @Test("invalid settings throw")
    func invalid() {
        let model = Self.twoLayer
        var params = PrintCurvesMorphParams(active: true)
        params.gammaFactorGreen = 0
        #expect(throws: SpektraError.self) {
            try PrintCurvesMorph.apply(
                logExposure: [0], model: model, params: params, positive: false)
        }
        params = PrintCurvesMorphParams(active: true)
        params.developerExhaustion = 1.5
        #expect(throws: SpektraError.self) {
            try PrintCurvesMorph.apply(
                logExposure: [0], model: model, params: params, positive: false)
        }
        #expect(throws: SpektraError.self) {
            try PrintCurvesMorph.apply(
                logExposure: [0], model: DensityCurvesModel(),
                params: PrintCurvesMorphParams(active: true), positive: false)
        }
    }

    @Test("a render with the morph on matches the oracle")
    func render() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true
        params.printRender.densityCurvesMorph = PrintCurvesMorphParams(active: true)
        params.printRender.densityCurvesMorph.gammaFactor = 1.25
        params.printRender.densityCurvesMorph.developerExhaustion = 0.4
        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let out = try Simulator(params).process(input)
        try expectParity(out.values, matches: "morph_render_portra400_endura")
    }
}
