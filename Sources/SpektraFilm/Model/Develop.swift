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

        let density = DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: normalised, axis: data.logExposure,
            gammaFactor: gammaFactor)

        // `consume` throughout: each of these takes the frame it is handed and writes over it, which
        // it can only do while nothing else holds a reference.
        let developed = Couplers.applyDensityCorrection(
            density: consume density,
            logRaw: logRaw,
            pixelSizeMicrons: pixelSizeMicrons,
            logExposure: data.logExposure,
            curves: normalised,
            params: dirCouplers,
            positive: profile.isPositive,
            gammaFactor: gammaFactor,
            spatial: spatial
        )

        guard let pixelSizeMicrons else { return developed }

        return Grain.apply(
            consume developed,
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
        let curves = try PrintCurvesMorph.apply(
            logExposure: data.logExposure,
            model: data.densityCurvesModel,
            params: morph,
            positive: profile.isPositive
        )
        return DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: curves, axis: data.logExposure, gammaFactor: 1.0)
    }
}

/// The print paper's characteristic curves, evaluated from their parametric fit.
///
/// Ports `utils/morph_curves.apply_print_curves_morph`. The printing stage always goes through here,
/// including when the morph is off: with `active: false` the curves come from
/// `_evaluate_fitted_density`, a sum of scaled normal CDFs over the emulsion's layers, and NOT from
/// `profile.data.densityCurves`. Reading the tabulated curves instead costs 3.2e-3 on the rendered
/// output, because the fit and the table are not the same function.
///
/// The active morph needs a Brent solve per control point to re-place the layer centres.
/// `Core/RootFind.swift` has the solver; the morph itself is not written, and it defaults off.
public enum PrintCurvesMorph {
    public static func apply(
        logExposure: [Double],
        model: DensityCurvesModel,
        params: PrintCurvesMorphParams,
        positive: Bool
    ) throws -> [Double] {
        guard !params.active else {
            throw SpektraError.unsupportedSetting(
                "print_render.density_curves_morph.active", value: "true")
        }
        return fittedCurves(logExposure: logExposure, model: model, positive: positive)
    }

    /// `_evaluate_fitted_density`.
    ///
    /// Each channel is a sum over layers of `amplitude * cdf((x - centre) / sigma)`. Positive stocks
    /// negate the argument, since their density falls with exposure.
    public static func fittedCurves(
        logExposure: [Double], model: DensityCurvesModel, positive: Bool
    ) -> [Double] {
        precondition(!model.isEmpty, "the print profile carries no fitted density_curves_model")
        let channels = model.channelCount
        var out = [Double](repeating: 0, count: logExposure.count * channels)
        for c in 0..<channels {
            for (i, x) in logExposure.enumerated() {
                var total = 0.0
                for layer in 0..<model.layerCount {
                    let z =
                        (x - model.center(channel: c, layer: layer))
                        / model.sigma(channel: c, layer: layer)
                    total +=
                        model.amplitude(channel: c, layer: layer)
                        * Erf.normalCDF(positive ? -z : z)
                }
                out[i * channels + c] = total
            }
        }
        return out
    }
}
