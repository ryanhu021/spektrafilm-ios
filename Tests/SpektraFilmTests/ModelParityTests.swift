import Testing

@testable import SpektraFilm

/// Checks the emulsion model against the reference.
///
/// Four stocks, two negative and two positive. The coupler inversion takes a different path for
/// positive film, and that path contains the non-monotonic interpolation.
@Suite("Emulsion model parity")
struct ModelParityTests {

    static let stocks = [
        "kodak_portra_400", "kodak_vision3_500t", "fujifilm_velvia_100", "kodak_ektachrome_100",
    ]

    // MARK: - DIR couplers

    @Test("the inhibition matrix matches, with default and tuned parameters")
    func inhibitionMatrix() throws {
        let defaults = Couplers.inhibitionMatrix(DirCouplersParams())
        try expectParity(flatten(defaults), matches: "couplers_matrix_default")

        var tuned = DirCouplersParams()
        tuned.inhibitionSameLayer = 0.6
        tuned.inhibitionInterlayer = 1.4
        tuned.gammaSameLayerRGB = (0.4, 0.3, 0.2)
        tuned.gammaInterlayerRedToGreenBlue = (0.2, 0.1)
        tuned.gammaInterlayerGreenToRedBlue = (0.3, 0.15)
        tuned.gammaInterlayerBlueToRedGreen = (0.25, 0.35)
        try expectParity(flatten(Couplers.inhibitionMatrix(tuned)), matches: "couplers_matrix_tuned")
    }

    @Test("curves before the couplers acted", arguments: stocks, ["default", "tuned"])
    func curvesBeforeCouplers(stock: String, variant: String) throws {
        let profile = try ProfileLibrary.load(stock)
        let normalised = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)

        var params = DirCouplersParams()
        if variant == "tuned" {
            params.inhibitionSameLayer = 0.6
            params.inhibitionInterlayer = 1.4
            params.gammaSameLayerRGB = (0.4, 0.3, 0.2)
            params.gammaInterlayerRedToGreenBlue = (0.2, 0.1)
            params.gammaInterlayerGreenToRedBlue = (0.3, 0.15)
            params.gammaInterlayerBlueToRedGreen = (0.25, 0.35)
        }
        let matrix = scaled(Couplers.inhibitionMatrix(params), by: params.amount)

        let result = Couplers.curvesBeforeCouplers(
            curves: normalised,
            logExposure: profile.data.logExposure,
            matrix: matrix,
            positive: profile.isPositive
        )
        try expectParity(result, matches: "couplers_before_\(stock)_\(variant)")
    }

    // MARK: - Density curves

    @Test("normalising the curves matches nanmin subtraction", arguments: stocks)
    func normalisedCurves(stock: String) throws {
        let profile = try ProfileLibrary.load(stock)
        let normalised = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)
        try expectParity(normalised, matches: "density_curves_normalised_\(stock)")
    }

    @Test("exposure to density, one gamma for all channels", arguments: stocks)
    func densityFromExposureScalarGamma(stock: String) throws {
        let profile = try ProfileLibrary.load(stock)
        let normalised = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)
        let result = DensityCurves.densityFromLogExposure(
            logExposure: try Golden("density_log_raw_input").imageBuffer(),
            curves: normalised,
            axis: profile.data.logExposure,
            gammaFactor: 1.0
        )
        try expectParity(result.values, matches: "density_from_exposure_\(stock)_gamma1")
    }

    @Test("exposure to density, per-channel gamma", arguments: stocks)
    func densityFromExposurePerChannelGamma(stock: String) throws {
        let profile = try ProfileLibrary.load(stock)
        let normalised = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)
        let result = DensityCurves.densityFromLogExposure(
            logExposure: try Golden("density_log_raw_input").imageBuffer(),
            curves: normalised,
            axis: profile.data.logExposure,
            gammaFactor: (0.85, 1.0, 1.2)
        )
        try expectParity(result.values, matches: "density_from_exposure_\(stock)_gamma_rgb")
    }

    // MARK: - Spectral products

    @Test("CMY density expands to a spectrum, with and without the base density")
    func spectralDensity() throws {
        let cmy = try Golden("spectral_density_cmy_input").imageBuffer()
        let channelDensity = try Golden("spectral_channel_density").values
        let baseDensity = try Golden("spectral_base_density").values

        let withBase = DensityCurves.spectralDensity(
            cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity)
        try expectParity(withBase.values, matches: "spectral_density_with_base")

        let withoutBase = DensityCurves.spectralDensity(
            cmy: cmy, channelDensity: channelDensity, baseDensity: nil)
        try expectParity(withoutBase.values, matches: "spectral_density_without_base")
    }

    /// `channel_density` and `base_density` both contain NaN for wavelengths the datasheet skips,
    /// so the spectral density inherits it and `density_to_light` turns it into zero. If NaN were
    /// dropped earlier, the unmeasured bands would contribute light that is not there.
    @Test("density to light zeroes NaN and scales by the illuminant")
    func densityToLight() throws {
        let density = try Golden("spectral_density_with_base").imageBuffer()
        let illuminant = try Illuminant(label: "D50").spectrum
        let light = DensityCurves.densityToLight(density, illuminant: illuminant)
        try expectParity(light.values, matches: "spectral_light_d50")
        #expect(light.values.allSatisfy { !$0.isNaN }, "density_to_light should leave no NaN")
        #expect(density.values.contains { $0.isNaN }, "the input should contain NaN to be a real test")
    }

    @Test("projecting spectral light onto the observer gives XYZ")
    func projectToXYZ() throws {
        let light = try Golden("spectral_light_d50").imageBuffer()
        let illuminant = try Illuminant(label: "D50").spectrum
        let normalisation = Observer.luminanceNormalisation(illuminant: illuminant)
        let xyz = DensityCurves.project(
            light, onto: Observer.cmfs, scale: 1.0 / normalisation)
        try expectParity(xyz.values, matches: "spectral_xyz_d50")
    }

    /// The spectral expansion turns 3 channels into 81, so a full frame cannot be converted in one
    /// allocation. Converting in row bands must give the same result as converting the whole frame.
    @Test("row-band conversion matches a single-shot conversion")
    func bandedConversionIsExact() throws {
        let cmy = try Golden("spectral_density_cmy_input").imageBuffer()
        let channelDensity = try Golden("spectral_channel_density").values
        let baseDensity = try Golden("spectral_base_density").values

        let whole = DensityCurves.spectralDensity(
            cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity)

        for bandRows in [1, 2, 5, 7, cmy.height, cmy.height + 3] {
            let banded = cmy.mapPerPixel(
                bandRows: bandRows, channelsOut: ColourTables.wavelengthCount
            ) { band in
                DensityCurves.spectralDensity(
                    cmy: band, channelDensity: channelDensity, baseDensity: baseDensity)
            }
            let report = parity(banded.values, whole.values)
            #expect(
                report.maxAbsolute == 0 && report.nanMismatches == 0,
                "bandRows=\(bandRows): \(report)")
        }
    }

    // MARK: - Helpers

    private func flatten(_ m: Matrix3) -> [Double] {
        [m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22]
    }

    private func scaled(_ m: Matrix3, by k: Double) -> Matrix3 {
        Matrix3(
            m.m00 * k, m.m01 * k, m.m02 * k,
            m.m10 * k, m.m11 * k, m.m12 * k,
            m.m20 * k, m.m21 * k, m.m22 * k)
    }
}
