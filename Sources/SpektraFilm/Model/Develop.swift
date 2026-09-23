import Foundation

/// Chemical development: log exposure in, dye density out.
///
/// Ports `model/develop.py`. Three steps in order, and the order matters because each reads the
/// previous one's output: the characteristic curves, the DIR couplers, then grain.
public enum Develop {

    /// `develop_simple`. Curves only, no couplers and no grain.
    ///
    /// Used for the black, white and midgray references the print balance depends on, where adding
    /// couplers or grain would put noise into a calibration constant.
    public static func simple(
        logRaw: ImageBuffer,
        logExposure: [Double],
        curves: [Double],
        gammaFactor: Double = 1.0
    ) -> ImageBuffer {
        DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: curves, axis: logExposure, gammaFactor: gammaFactor)
    }

    /// `develop`, the film path.
    ///
    /// The curves are fog-normalised first, so they start at zero density and the grain model's
    /// `densityMin` sets the real floor.
    ///
    /// - Parameters:
    ///   - pixelSizeMicrons: `nil` when the pipeline was injected past preprocessing, as a LUT bake
    ///     does. The coupler stage handles that by skipping its spatial term; grain needs a pitch,
    ///     so it is skipped entirely.
    public static func film(
        logRaw: ImageBuffer,
        pixelSizeMicrons: Double?,
        profile: Profile,
        dirCouplers: DirCouplersParams,
        grain: GrainParams,
        gammaFactor: Double = 1.0,
        grainSeed: UInt64 = 0,
        bypassGrain: Bool = false,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        let data = profile.data
        let normalised = DensityCurves.normalized(
            curves: data.densityCurves, minima: data.densityCurveMinima)

        var density = DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: normalised, axis: data.logExposure,
            gammaFactor: gammaFactor)

        density = Couplers.applyDensityCorrection(
            density: density,
            logRaw: logRaw,
            pixelSizeMicrons: pixelSizeMicrons,
            logExposure: data.logExposure,
            curves: normalised,
            params: dirCouplers,
            positive: profile.isPositive,
            gammaFactor: gammaFactor,
            spatial: spatial
        )

        guard let pixelSizeMicrons else { return density }

        return Grain.apply(
            density,
            pixelSizeMicrons: pixelSizeMicrons,
            params: grain,
            densityCurves: normalised,
            densityCurvesLayers: data.densityCurvesLayers,
            positive: profile.isPositive,
            seed: grainSeed,
            bypass: bypassGrain,
            spatial: spatial
        )
    }

    /// `develop_print_morph`, the print path.
    ///
    /// Print paper has no couplers modelled and no grain: it does not sample a scene, so it is
    /// designed with little channel cross-talk to begin with. The creative curve morph is off by
    /// default.
    public static func print(
        logRaw: ImageBuffer,
        profile: Profile,
        morph: PrintCurvesMorphParams
    ) throws -> ImageBuffer {
        let data = profile.data
        let curves: [Double]
        if morph.active {
            curves = try PrintCurvesMorph.apply(
                logExposure: data.logExposure,
                model: data.densityCurvesModel,
                params: morph,
                positive: profile.isPositive
            )
        } else {
            curves = data.densityCurves
        }
        return DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: curves, axis: data.logExposure, gammaFactor: 1.0)
    }
}

/// Creative reshaping of the print paper's characteristic curves.
///
/// Off by default (`PrintRenderingParams` constructs it with `active: false`), so nothing on the
/// default render path reaches it. Porting `utils/morph_curves.py` needs the parametric CDF curve
/// fit and a Brent solve per control point; `Core/RootFind.swift` has the solver, the fit is not
/// written yet.
public enum PrintCurvesMorph {
    public static func apply(
        logExposure: [Double],
        model: DensityCurvesModel,
        params: PrintCurvesMorphParams,
        positive: Bool
    ) throws -> [Double] {
        throw SpektraError.unsupportedSetting(
            "print_render.density_curves_morph.active", value: "true")
    }
}
