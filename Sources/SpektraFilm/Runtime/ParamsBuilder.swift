import Foundation

/// Turns a freshly constructed parameter set into the one the pipeline actually runs.
///
/// Ports `runtime/params_builder.digest_params`. The dataclass defaults in ``RuntimePhotoParams``
/// are not what renders: this layer overrides the enlarger's neutral filter positions from a
/// measured database, seeds the coupler and halation parameters from the stock's own tags, and
/// applies the debug switches. Skipping it changes the render, so ``Simulator`` digests on the way
/// in and there is no way to construct a pipeline around undigested params.
public enum ParamsBuilder {

    /// `digest_params`.
    public static func digest(
        _ params: RuntimePhotoParams, applyStockSpecifics: Bool = true
    ) throws -> RuntimePhotoParams {
        var p = params
        try applyDatabaseNeutralPrintFilters(&p)

        if p.settings.previewMode { applyPreviewMode(&p) }

        if applyStockSpecifics {
            applyFilmSpecifics(&p)
            applyPrintSpecifics(&p)
        }

        applyDebugSwitches(&p)
        return p
    }

    // MARK: - Neutral print filters

    /// `apply_database_neutral_print_filters`.
    ///
    /// The filter positions that make an 18% grey card neutral depend on the paper, the enlarger
    /// lamp and the film together, so they are measured per combination rather than derived. The
    /// dataclass defaults of 0 / 65 / 55 are a fallback for combinations the database does not cover.
    static func applyDatabaseNeutralPrintFilters(_ p: inout RuntimePhotoParams) throws {
        guard p.settings.neutralPrintFiltersFromDatabase else { return }
        guard
            let entry = try NeutralPrintFilters.shared.lookup(
                paper: p.print.info.stock,
                illuminant: p.enlarger.illuminant,
                film: p.film.info.stock)
        else { return }
        p.enlarger.cFilterNeutral = entry.c
        p.enlarger.mFilterNeutral = entry.m
        p.enlarger.yFilterNeutral = entry.y
    }

    // MARK: - Preview mode

    /// Drops the expensive spatial and stochastic work while keeping the scatter and halation kernel
    /// sigmas, which the reference preserves deliberately.
    static func applyPreviewMode(_ p: inout RuntimePhotoParams) {
        p.enlarger.lensBlur = 0
        p.filmRender.dirCouplers.diffusionSizeMicrons = 0
        p.filmRender.grain.active = false
        p.filmRender.grain.particleAreaMicronsSquared = 0
        p.filmRender.grain.blur = 0
        p.printRender.glare.blur = 0
        p.camera.lensBlurMicrons = 0
        p.scanner.lensBlur = 0
        p.scanner.unsharpMask = (0, 0)
    }

    // MARK: - Stock specifics

    /// `_apply_film_specifics`.
    ///
    /// The coupler gammas here are fitted, and they differ from the dataclass defaults. Positive
    /// stocks get much weaker inhibition than negatives, and Velvia and Provia are tuned again on
    /// top of that.
    static func applyFilmSpecifics(_ p: inout RuntimePhotoParams) {
        if p.film.isPositive {
            p.filmRender.dirCouplers.gammaSameLayerRGB = (0.12, 0.08, 0.06)
            p.filmRender.dirCouplers.gammaInterlayerRedToGreenBlue = (0.12, 0.06)
            p.filmRender.dirCouplers.gammaInterlayerGreenToRedBlue = (0.08, 0.06)
            p.filmRender.dirCouplers.gammaInterlayerBlueToRedGreen = (0.06, 0.06)
        }
        if p.film.isNegative {
            p.filmRender.dirCouplers.gammaSameLayerRGB = (0.336, 0.319, 0.273)
            p.filmRender.dirCouplers.gammaInterlayerRedToGreenBlue = (0.353, 0.302)
            p.filmRender.dirCouplers.gammaInterlayerGreenToRedBlue = (0.154, 0.353)
            p.filmRender.dirCouplers.gammaInterlayerBlueToRedGreen = (0.168, 0.226)
        }

        applyHalationPreset(&p)

        switch p.film.info.stock {
        case "fujifilm_velvia_100":
            p.filmRender.dirCouplers.gammaSameLayerRGB = (0.108, 0.072, 0.054)
            p.filmRender.dirCouplers.gammaInterlayerRedToGreenBlue = (0.108, 0.054)
            p.filmRender.dirCouplers.gammaInterlayerGreenToRedBlue = (0.072, 0.054)
            p.filmRender.dirCouplers.gammaInterlayerBlueToRedGreen = (0.054, 0.054)
        case "fujifilm_provia_100f":
            p.filmRender.dirCouplers.gammaSameLayerRGB = (0.156, 0.104, 0.078)
            p.filmRender.dirCouplers.gammaInterlayerRedToGreenBlue = (0.156, 0.078)
            p.filmRender.dirCouplers.gammaInterlayerGreenToRedBlue = (0.104, 0.078)
            p.filmRender.dirCouplers.gammaInterlayerBlueToRedGreen = (0.078, 0.078)
        default:
            break
        }
    }

    /// `_apply_print_specifics`. Empty upstream, kept so the call site stays visible.
    static func applyPrintSpecifics(_ p: inout RuntimePhotoParams) {}

    /// Halation baselines keyed by the profile's own `use` and `antihalation` tags.
    ///
    /// The base material sets the blur width: still film is triacetate at 120 to 140 microns, giving
    /// sigma around 65; cine film is PET at 95 to 125 microns, giving around 50. The antihalation
    /// layer sets the strength. The user-facing amount and scale knobs stay at 1.0 on top of these.
    static let halationPresets:
        [String: (sigma: (Double, Double, Double), strength: (Double, Double, Double))] = [
            "still/strong": ((65.0, 65.0, 65.0), (0.015, 0.005, 0.0)),
            "still/weak": ((65.0, 65.0, 65.0), (0.08, 0.02, 0.0)),
            "still/no": ((65.0, 65.0, 65.0), (0.30, 0.10, 0.015)),
            "cine/strong": ((50.0, 50.0, 50.0), (0.015, 0.005, 0.0)),
            "cine/weak": ((50.0, 50.0, 50.0), (0.08, 0.02, 0.0)),
            "cine/no": ((50.0, 50.0, 50.0), (0.30, 0.10, 0.015)),
        ]

    static func applyHalationPreset(_ p: inout RuntimePhotoParams) {
        guard p.film.isFilm else { return }
        let key = "\(p.film.info.use.rawValue)/\(p.film.info.antihalation.rawValue)"
        guard let preset = halationPresets[key] else { return }
        p.filmRender.halation.halationFirstSigmaMicrons = preset.sigma
        p.filmRender.halation.halationStrength = preset.strength
    }

    // MARK: - Debug switches

    /// The `lut_mode`, `deactivate_spatial_effects` and `deactivate_stochastic_effects` cascade.
    ///
    /// Order matters: `lut_mode` promotes the other two, so it runs first.
    static func applyDebugSwitches(_ p: inout RuntimePhotoParams) {
        if p.debug.lutMode {
            p.debug.deactivateSpatialEffects = true
            p.debug.deactivateStochasticEffects = true
            p.camera.autoExposure = false
            p.camera.exposureCompensationEV = 0.0
            p.enlarger.printExposureCompensation = false
            p.enlarger.printExposure = 1.0
            // The highlight boost normalises by the image-wide maximum, so the same input value maps
            // to different outputs depending on the rest of the frame. A static LUT cannot represent
            // that, so it goes off with auto-exposure.
            p.filmRender.halation.boostEV = 0.0
            p.scanner.whiteCorrection = false
            p.scanner.blackCorrection = false
            p.scanner.unsharpMask = (0, 0)
        }

        if p.debug.deactivateSpatialEffects {
            // Halation is entirely spatial. Both the active flag and the kernel widths are cleared,
            // so it stays inert even if a width-independent term is added later.
            p.filmRender.halation.active = false
            p.filmRender.halation.scatterCoreMicrons = (0, 0, 0)
            p.filmRender.halation.scatterTailMicrons = (0, 0, 0)
            p.filmRender.halation.halationFirstSigmaMicrons = (0, 0, 0)
            p.filmRender.dirCouplers.diffusionSizeMicrons = 0
            p.filmRender.grain.blur = 0
            p.filmRender.grain.blurDyeCloudsMicrons = 0
            p.printRender.glare.blur = 0
            p.camera.lensBlurMicrons = 0
            p.enlarger.lensBlur = 0
            p.enlarger.diffusionFilter.active = false
            p.camera.diffusionFilter.active = false
            p.scanner.lensBlur = 0
            p.scanner.unsharpMask = (0, 0)
        }

        if p.debug.deactivateStochasticEffects {
            p.filmRender.grain.active = false
            p.printRender.glare.active = false
        }
    }
}

/// The measured neutral enlarger filter positions, keyed by paper, lamp and film.
///
/// Loaded once from `neutral_print_filters.json`. A combination the database does not cover keeps
/// the dataclass defaults, which is what the reference does after printing a warning.
public final class NeutralPrintFilters: @unchecked Sendable {
    public static let shared = NeutralPrintFilters()

    private let lock = NSLock()
    private var table: [String: [String: [String: [Double]]]]?

    private init() {}

    public func lookup(
        paper: String, illuminant: String, film: String
    ) throws -> (
        c: Double, m: Double, y: Double
    )? {
        let loaded = try load()
        guard let values = loaded[paper]?[illuminant]?[film], values.count == 3 else { return nil }
        return (values[0], values[1], values[2])
    }

    private func load() throws -> [String: [String: [String: [Double]]]] {
        lock.lock()
        defer { lock.unlock() }
        if let table { return table }

        guard
            let url = Bundle.module.url(
                forResource: "neutral_print_filters", withExtension: "json",
                subdirectory: "Data/filters")
        else {
            throw SpektraError.missingResource("Data/filters/neutral_print_filters.json")
        }
        let decoded: [String: [String: [String: [Double]]]]
        do {
            decoded = try JSONDecoder().decode(
                [String: [String: [String: [Double]]]].self, from: try Data(contentsOf: url))
        } catch {
            throw SpektraError.malformedResource(
                "neutral_print_filters.json", reason: String(describing: error))
        }
        table = decoded
        return decoded
    }
}
