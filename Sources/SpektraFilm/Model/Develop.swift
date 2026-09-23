import Foundation

/// Chemical development: log exposure in, dye density out.
///
/// Ports `model/develop.py`. Three steps, in order: the characteristic curves, the DIR couplers,
/// then grain. Each reads the previous one's output.
public enum Develop {

    /// `develop_simple`. Curves only, no couplers and no grain.
    ///
    /// Used for the black, white and midgray references the print balance depends on. Couplers or
    /// grain would put noise into a calibration constant.
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
    ///     does. The coupler stage then skips its spatial term. Grain needs a pitch, so it is
    ///     skipped entirely.
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

        // `consume` throughout: each call writes over the frame it is handed, and can only do so
        // while nothing else holds a reference.
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
    /// Print paper has no modelled couplers and no grain. It does not sample a scene, so it is
    /// designed with little channel cross-talk. The creative curve morph is off by default.
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
/// even with the morph off. With `active: false` the curves come from `_evaluate_fitted_density`, a
/// sum of scaled normal CDFs over the emulsion's layers, and NOT from `profile.data.densityCurves`.
/// The fit and the table are different functions: reading the table moves the rendered output by
/// 3.2e-3.
///
/// With the morph on, each layer's centre and sigma are divided by an effective gamma chosen by the
/// layer's speed, and developer exhaustion blends every layer's CDF toward a matched Gumbel CDF. A
/// per-channel shift, found with one Brent solve, then puts the density at zero log exposure back
/// where it was.
public enum PrintCurvesMorph {
    /// `SIGMA_FLOOR`.
    static let sigmaFloor = 0.05
    /// `_GUMBEL_LOCATION` and `_GUMBEL_WIDTH`: the Gumbel CDF with the normal's median and slope at
    /// the median.
    static let gumbelLocation = -Foundation.log(Foundation.log(2.0))
    static let gumbelWidth = 0.5 * Foundation.log(2.0) * (2.0 * Double.pi).squareRoot()

    public static func apply(
        logExposure: [Double],
        model: DensityCurvesModel,
        params: PrintCurvesMorphParams,
        positive: Bool
    ) throws -> [Double] {
        guard params.active else {
            return fittedCurves(logExposure: logExposure, model: model, positive: positive)
        }
        guard !model.isEmpty else {
            throw SpektraError.unsupportedSetting(
                "print_render.density_curves_morph.active",
                value: "true, but the print profile has no fitted density_curves_model")
        }
        let factors: [(String, Double)] = [
            ("gamma_factor", params.gammaFactor),
            ("gamma_factor_fast", params.gammaFactorFast),
            ("gamma_factor_slow", params.gammaFactorSlow),
            ("gamma_factor_red", params.gammaFactorRed),
            ("gamma_factor_green", params.gammaFactorGreen),
            ("gamma_factor_blue", params.gammaFactorBlue),
        ]
        for (name, value) in factors where !(value > 0) {
            throw SpektraError.unsupportedSetting(
                "print_render.density_curves_morph.\(name)", value: "\(value), must be > 0")
        }
        guard params.developerExhaustion >= 0, params.developerExhaustion <= 1 else {
            throw SpektraError.unsupportedSetting(
                "print_render.density_curves_morph.developer_exhaustion",
                value: "\(params.developerExhaustion), must be in [0, 1]")
        }

        let channels = model.channelCount
        var out = [Double](repeating: 0, count: logExposure.count * channels)
        for c in 0..<channels {
            let layer = try morphedChannel(model, params, channel: c, positive: positive)
            for (i, x) in logExposure.enumerated() {
                out[i * channels + c] = channelDensity(x, layer, positive: positive)
            }
        }
        return out
    }

    /// One channel's layers after the morph: `_morph_channel_params`.
    struct Layers {
        var centers: [Double]
        var amplitudes: [Double]
        var sigmas: [Double]
        var gumbelMix: Double
    }

    static func morphedChannel(
        _ model: DensityCurvesModel, _ params: PrintCurvesMorphParams, channel c: Int,
        positive: Bool
    ) throws -> Layers {
        let n = model.layerCount
        var layers = Layers(
            centers: (0..<n).map { model.center(channel: c, layer: $0) },
            amplitudes: (0..<n).map { model.amplitude(channel: c, layer: $0) },
            sigmas: (0..<n).map { model.sigma(channel: c, layer: $0) },
            gumbelMix: params.developerExhaustion)

        // `_speed_layer_indices`: ascending centre, ties in index order as NumPy's small-array
        // argsort leaves them. With fewer than three layers the indices coincide, and the updates
        // below then compound on the same layer, as they do upstream. Layers beyond the fastest,
        // the middle and the slowest are not scaled at all.
        let order = (0..<n).sorted {
            layers.centers[$0] < layers.centers[$1]
                || (layers.centers[$0] == layers.centers[$1] && $0 < $1)
        }
        let fast = order[0]
        let mid = order[n / 2]
        let slow = order[n - 1]

        let channelFactor = [params.gammaFactorRed, params.gammaFactorGreen, params.gammaFactorBlue][c]
        let gFast = params.gammaFactor * channelFactor * params.gammaFactorFast
        let gMid = params.gammaFactor * channelFactor * params.gammaFactorSlow
        let gSlow = params.gammaFactor * channelFactor * params.gammaFactorSlow
        guard gFast > 0, gMid > 0, gSlow > 0 else {
            throw SpektraError.unsupportedSetting(
                "print_render.density_curves_morph",
                value: "effective gamma for channel \(c) is not positive")
        }
        for (index, gamma) in [(fast, gFast), (mid, gMid), (slow, gSlow)] {
            layers.sigmas[index] = Swift.max(layers.sigmas[index] / gamma, sigmaFloor)
            layers.centers[index] = layers.centers[index] / gamma
        }

        let offset = exhaustionCentreOffset(layers, positive: positive)
        for i in 0..<n { layers.centers[i] = layers.centers[i] + offset }
        return layers
    }

    /// `_developer_exhaustion_center_offset`: the shift that restores the density at zero log
    /// exposure once the Gumbel blend has skewed the layer CDFs.
    static func exhaustionCentreOffset(_ layers: Layers, positive: Bool) -> Double {
        // `np.allclose(mix, 0)`: absolute tolerance 1e-8.
        guard abs(layers.gumbelMix) > 1e-8 else { return 0 }
        var unmixed = layers
        unmixed.gumbelMix = 0
        let target = channelDensity(0, unmixed, positive: positive)

        func residual(_ offset: Double) -> Double {
            var shifted = layers
            for i in shifted.centers.indices { shifted.centers[i] = layers.centers[i] + offset }
            return channelDensity(0, shifted, positive: positive) - target
        }
        guard abs(residual(0)) > 1e-12 else { return 0 }
        return RootFind.expandingBracketRoot(xTolerance: 1e-10, residual) ?? 0
    }

    /// `_evaluate_channel_density` at one log exposure.
    static func channelDensity(_ x: Double, _ layers: Layers, positive: Bool) -> Double {
        var total = 0.0
        for i in layers.centers.indices {
            let z = (x - layers.centers[i]) / layers.sigmas[i]
            total += layers.amplitudes[i] * layerCDF(z, positive: positive, mix: layers.gumbelMix)
        }
        return total
    }

    /// `_layer_cdf`.
    static func layerCDF(_ z: Double, positive: Bool, mix: Double) -> Double {
        let signed = positive ? -z : z
        let cdf = Erf.normalCDF(signed)
        guard mix > 0 else { return cdf }
        let gumbel = Foundation.exp(-Foundation.exp(-(signed / gumbelWidth + gumbelLocation)))
        return (1.0 - mix) * cdf + mix * gumbel
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
