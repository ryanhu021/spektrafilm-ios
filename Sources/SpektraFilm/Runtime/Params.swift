import Foundation

// The full parameter surface, mirroring `spektrafilm.runtime.params_schema`. Every default matches
// the reference. The defaults define the render, so a wrong one shifts every photo. Field names
// follow Swift convention, and each group documents the Python name it came from.

/// `DiffusionFilterParams`. Models a screw-on diffusion filter.
public struct DiffusionFilterParams: Sendable, Equatable {
    /// PSF shape and absorption regime. Keys of `_DIFFUSION_FILTER_SHAPES` in `model/diffusion.py`.
    public enum Family: String, Sendable, CaseIterable {
        case glimmerglass
        case blackProMist = "black_pro_mist"
        case proMist = "pro_mist"
        case cinebloom
    }

    public var active: Bool = false
    public var family: Family = .blackProMist
    /// Commercial filter stops: 0, 1/8, 1/4, 1/2, 1, 2, interpolated in between.
    public var strength: Double = 0.5
    /// Multiplier on every image-plane PSF width.
    public var spatialScale: Double = 1.0
    /// Additive bias on the halo warmth axis. 0 uses the family default. Positive pushes warm light
    /// to the outer halo and cool light inward. Soft-clamped to [-1.5, 1.5].
    public var haloWarmth: Double = 0.0
    /// Per-group fine tuning. The three intensities scale the core, halo and bloom weights, which
    /// are then renormalised to sum to 1, so these redistribute energy without changing the total
    /// deflected fraction. The sizes scale each group's widths uniformly.
    public var coreIntensity: Double = 1.0
    public var coreSize: Double = 1.0
    public var haloIntensity: Double = 1.0
    public var haloSize: Double = 1.0
    public var bloomIntensity: Double = 1.0
    public var bloomSize: Double = 1.0

    public init() {}
}

/// `CameraParams`.
public struct CameraParams: Sendable, Equatable {
    public enum AutoExposureMethod: String, Sendable, CaseIterable {
        case centerWeighted = "center_weighted"
        case matrix
        case multiZone = "multi_zone"
        case partial
        case highlightWeighted = "highlight_weighted"
        case median
        case average
    }

    public var exposureCompensationEV: Double = 0.0
    public var autoExposure: Bool = true
    public var autoExposureMethod: AutoExposureMethod = .centerWeighted
    public var lensBlurMicrons: Double = 0.0
    /// Sets the pixel pitch, and through it every effect specified in microns.
    public var filmFormatMillimetres: Double = 35.0
    /// `(amplitude, wavelength_nm, width_nm)`. Amplitude 0 disables the filter.
    public var filterUV: (amplitude: Double, wavelength: Double, width: Double) = (0.0, 410.0, 8.0)
    public var filterIR: (amplitude: Double, wavelength: Double, width: Double) = (0.0, 675.0, 15.0)
    public var diffusionFilter = DiffusionFilterParams()

    public init() {}

    public static func == (a: CameraParams, b: CameraParams) -> Bool {
        a.exposureCompensationEV == b.exposureCompensationEV && a.autoExposure == b.autoExposure
            && a.autoExposureMethod == b.autoExposureMethod
            && a.lensBlurMicrons == b.lensBlurMicrons
            && a.filmFormatMillimetres == b.filmFormatMillimetres && a.filterUV == b.filterUV
            && a.filterIR == b.filterIR && a.diffusionFilter == b.diffusionFilter
    }
}

/// `EnlargerParams`. The colour head that projects the negative onto paper.
public struct EnlargerParams: Sendable, Equatable {
    public var illuminant: String = "TH-KG3"
    public var printExposure: Double = 1.0
    public var printExposureCompensation: Bool = true
    public var normalizePrintExposure: Bool = true
    /// Steps away from the neutral position of the yellow and magenta dichroic filters.
    public var yFilterShift: Double = 0.0
    public var mFilterShift: Double = 0.0
    /// Neutral filter positions in Kodak CC units, where 100 units is 1.0 density.
    public var yFilterNeutral: Double = 55
    public var mFilterNeutral: Double = 65
    public var cFilterNeutral: Double = 0
    public var lensBlur: Double = 0.0
    public var diffusionFilter = DiffusionFilterParams()
    /// Pre-flashing the paper holds highlights. 0 disables it.
    public var preflashExposure: Double = 0.0
    public var preflashYFilterShift: Double = 0.0
    public var preflashMFilterShift: Double = 0.0

    public init() {}
}

/// `ScannerParams`.
public struct ScannerParams: Sendable, Equatable {
    public var lensBlur: Double = 0.0
    public var whiteCorrection: Bool = false
    public var blackCorrection: Bool = false
    public var whiteLevel: Double = 0.98
    public var blackLevel: Double = 0.01
    /// `(sigma, amount)`. Both must be positive for the unsharp mask to run.
    public var unsharpMask: (sigma: Double, amount: Double) = (0.7, 0.7)

    public init() {}

    public static func == (a: ScannerParams, b: ScannerParams) -> Bool {
        a.lensBlur == b.lensBlur && a.whiteCorrection == b.whiteCorrection
            && a.blackCorrection == b.blackCorrection && a.whiteLevel == b.whiteLevel
            && a.blackLevel == b.blackLevel && a.unsharpMask == b.unsharpMask
    }
}

/// `GrainParams`. The stochastic silver-halide particle model.
public struct GrainParams: Sendable, Equatable {
    public var active: Bool = true
    public var sublayersActive: Bool = true
    public var particleAreaMicronsSquared: Double = 0.2
    public var particleScale: (Double, Double, Double) = (1.6, 1.6, 3.2)
    public var particleScaleLayers: (Double, Double, Double) = (2.0, 1.0, 0.5)
    public var densityMin: (Double, Double, Double) = (0.03, 0.03, 0.03)
    public var uniformity: (Double, Double, Double) = (0.97, 0.99, 0.97)
    public var blur: Double = 0.65
    public var blurDyeCloudsMicrons: Double = 1.0
    public var microStructure: (Double, Double) = (0.2, 30)
    public var subLayerCount: Int = 1

    public init() {}

    public static func == (a: GrainParams, b: GrainParams) -> Bool {
        a.active == b.active && a.sublayersActive == b.sublayersActive
            && a.particleAreaMicronsSquared == b.particleAreaMicronsSquared
            && a.particleScale == b.particleScale
            && a.particleScaleLayers == b.particleScaleLayers && a.densityMin == b.densityMin
            && a.uniformity == b.uniformity && a.blur == b.blur
            && a.blurDyeCloudsMicrons == b.blurDyeCloudsMicrons
            && a.microStructure == b.microStructure && a.subLayerCount == b.subLayerCount
    }
}

/// `HalationParams`. In-emulsion scatter plus back-reflection from the film base.
public struct HalationParams: Sendable, Equatable {
    public var active: Bool = true
    /// High-level scalars. 1.0 leaves the physical low-level defaults alone.
    public var scatterAmount: Double = 1.0
    public var scatterSpatialScale: Double = 1.0
    public var halationAmount: Double = 1.0
    public var halationSpatialScale: Double = 1.0
    /// In-emulsion scatter, an energy-preserving Gaussian core plus an exponential tail.
    /// `scatterTailMicrons` is the exponential decay constant, applied internally as a Gaussian
    /// mixture.
    public var scatterCoreMicrons: (Double, Double, Double) = (2.2, 2.0, 1.6)
    public var scatterTailMicrons: (Double, Double, Double) = (9.3, 9.7, 9.1)
    public var scatterTailWeight: (Double, Double, Double) = (0.78, 0.65, 0.67)
    /// Highlight boost, which reconstructs pre-clip irradiance before propagation.
    public var boostEV: Double = 0.0
    public var boostRange: Double = 0.3
    public var protectEV: Double = 4.0
    /// Back-reflection halation, an additive sum of N Gaussians with sqrt(k) widths.
    public var halationStrength: (Double, Double, Double) = (0.05, 0.015, 0.0)
    public var halationFirstSigmaMicrons: (Double, Double, Double) = (65.0, 65.0, 65.0)
    public var halationBounceCount: Int = 3
    public var halationBounceDecay: Double = 0.5
    public var halationRenormalize: Bool = true

    public init() {}

    public static func == (a: HalationParams, b: HalationParams) -> Bool {
        a.active == b.active && a.scatterAmount == b.scatterAmount
            && a.scatterSpatialScale == b.scatterSpatialScale
            && a.halationAmount == b.halationAmount
            && a.halationSpatialScale == b.halationSpatialScale
            && a.scatterCoreMicrons == b.scatterCoreMicrons
            && a.scatterTailMicrons == b.scatterTailMicrons
            && a.scatterTailWeight == b.scatterTailWeight && a.boostEV == b.boostEV
            && a.boostRange == b.boostRange && a.protectEV == b.protectEV
            && a.halationStrength == b.halationStrength
            && a.halationFirstSigmaMicrons == b.halationFirstSigmaMicrons
            && a.halationBounceCount == b.halationBounceCount
            && a.halationBounceDecay == b.halationBounceDecay
            && a.halationRenormalize == b.halationRenormalize
    }
}

/// `DirCouplersParams`. Development-inhibitor-releasing couplers, which raise saturation, contrast
/// and local contrast.
public struct DirCouplersParams: Sendable, Equatable {
    public var active: Bool = true
    public var amount: Double = 1.0
    public var inhibitionSameLayer: Double = 1.0
    public var inhibitionInterlayer: Double = 1.0
    /// Diagonal of the inhibition matrix: how much each layer inhibits itself.
    public var gammaSameLayerRGB: (Double, Double, Double) = (0.341, 0.324, 0.273)
    /// Off-diagonal terms, donor to receiver.
    public var gammaInterlayerRedToGreenBlue: (Double, Double) = (0.355, 0.305)
    public var gammaInterlayerGreenToRedBlue: (Double, Double) = (0.154, 0.358)
    public var gammaInterlayerBlueToRedGreen: (Double, Double) = (0.171, 0.225)
    public var diffusionSizeMicrons: Double = 20.0
    /// Exponential tail for Levy-like processes or environmental heterogeneity.
    public var diffusionTailMicrons: Double = 200.0
    public var diffusionTailWeight: Double = 0.06

    public init() {}

    public static func == (a: DirCouplersParams, b: DirCouplersParams) -> Bool {
        a.active == b.active && a.amount == b.amount
            && a.inhibitionSameLayer == b.inhibitionSameLayer
            && a.inhibitionInterlayer == b.inhibitionInterlayer
            && a.gammaSameLayerRGB == b.gammaSameLayerRGB
            && a.gammaInterlayerRedToGreenBlue == b.gammaInterlayerRedToGreenBlue
            && a.gammaInterlayerGreenToRedBlue == b.gammaInterlayerGreenToRedBlue
            && a.gammaInterlayerBlueToRedGreen == b.gammaInterlayerBlueToRedGreen
            && a.diffusionSizeMicrons == b.diffusionSizeMicrons
            && a.diffusionTailMicrons == b.diffusionTailMicrons
            && a.diffusionTailWeight == b.diffusionTailWeight
    }
}

/// `GlareParams`. Stray light in the scanner or the viewing setup.
public struct GlareParams: Sendable, Equatable {
    public var active: Bool = true
    public var percent: Double = 0.03
    public var roughness: Double = 0.7
    public var blur: Double = 0.5

    public init() {}
}

/// `PrintCurvesMorphParams` from `utils/morph_curves.py`. Creative reshaping of the paper's
/// characteristic curves. `PrintRenderingParams` turns this off by default.
public struct PrintCurvesMorphParams: Sendable, Equatable {
    public var active: Bool = true
    public var gammaFactor: Double = 1.0
    public var gammaFactorFast: Double = 1.0
    public var gammaFactorSlow: Double = 1.0
    public var gammaFactorRed: Double = 1.0
    public var gammaFactorGreen: Double = 1.0
    public var gammaFactorBlue: Double = 1.0
    public var developerExhaustion: Double = 0.0

    public init(active: Bool = true) {
        self.active = active
    }
}

/// `InputGamutCompressSpec`. Pulls input chromaticities inside the visible spectral locus, where
/// Hanatos-2025 spectral upsampling is defined. Baked into the per-film LUT at build time.
public struct InputGamutCompressSpec: Sendable, Equatable {
    public enum Algorithm: String, Sendable, CaseIterable {
        /// Radial compression in CIE 1931 chromaticity, from the film reference illuminant toward
        /// the spectral locus. The production default.
        case xy
        /// Chroma reduction at constant Oklch lightness and hue.
        case oklch
    }

    public var active: Bool = true
    public var algorithm: Algorithm = .xy
    /// Reinhard knee `(threshold, limit, power)`:
    /// `d' = t + s * n / (1 + n^p)^(1/p)` with `n = (d - t) / s` and `s = limit - t`.
    public var knee: (threshold: Double, limit: Double, power: Double) = (0.0, 1.0, 6.0)

    public init() {}

    public func validate() throws {
        guard knee.threshold >= 0, knee.threshold < 1 else {
            throw SpektraError.unsupportedSetting(
                "input_gamut_compress.knee.threshold", value: "\(knee.threshold)")
        }
        guard knee.limit > 0 else {
            throw SpektraError.unsupportedSetting(
                "input_gamut_compress.knee.limit", value: "\(knee.limit)")
        }
        guard knee.power > 0 else {
            throw SpektraError.unsupportedSetting(
                "input_gamut_compress.knee.power", value: "\(knee.power)")
        }
    }

    public static func == (a: InputGamutCompressSpec, b: InputGamutCompressSpec) -> Bool {
        a.active == b.active && a.algorithm == b.algorithm && a.knee == b.knee
    }
}

/// `OutputGamutCompressSpec`. Compresses out-of-gamut chromaticities into the output primaries cube,
/// and with `lightnessCompression` also pulls above-white pixels back in. With both enabled, the
/// default `cam16ucs` keeps the output in [0, 1] without a downstream clip. The other perceptual
/// algorithms overshoot slightly, up to 1.00675 for `jzazbz`.
public struct OutputGamutCompressSpec: Sendable, Equatable {
    public enum Algorithm: String, Sendable, CaseIterable {
        /// Passes output RGB through. Nothing else clips in the runtime, so the result can leave
        /// [0, 1].
        case off
        case acesRGC = "aces_rgc"
        case oklch
        case oklrab
        case jzazbz
        /// CIECAM16 uniform colour space, the default. Models chromatic adaptation and viewing
        /// conditions, fixed at L_A = 64 cd/m^2, Y_b = 20, average surround.
        case cam16ucs
    }

    public var algorithm: Algorithm = .cam16ucs
    public var knee: (threshold: Double, limit: Double, power: Double) = (0.0, 1.0, 6.0)
    /// One-sided soft roll-off on the perceptual lightness axis. Black stays at 0. `nil` disables
    /// it, which lets super-bright pixels through.
    public var lightnessCompression: (threshold: Double, limit: Double, power: Double)? = (
        0.7, 1.0, 2.2
    )

    public init() {}

    public static func == (a: OutputGamutCompressSpec, b: OutputGamutCompressSpec) -> Bool {
        a.algorithm == b.algorithm && a.knee == b.knee
            && a.lightnessCompression?.threshold == b.lightnessCompression?.threshold
            && a.lightnessCompression?.limit == b.lightnessCompression?.limit
            && a.lightnessCompression?.power == b.lightnessCompression?.power
    }
}

/// `FilmRenderingParams`. How the negative is rendered, as opposed to what the stock measures.
public struct FilmRenderingParams: Sendable, Equatable {
    public var densityCurveGamma: Double = 1.0
    public var grain = GrainParams()
    public var halation = HalationParams()
    public var dirCouplers = DirCouplersParams()
    public var glare = GlareParams()

    public init() {}
}

/// `PrintRenderingParams`.
public struct PrintRenderingParams: Sendable, Equatable {
    public var glare = GlareParams()
    /// Off by default, matching the reference's `PrintCurvesMorphParams(active=False)`.
    public var densityCurvesMorph = PrintCurvesMorphParams(active: false)

    public init() {}
}

/// `IOParams`. What comes in, what goes out, and the crop.
public struct IOParams: Sendable, Equatable {
    public var inputColourSpace: String = "ProPhoto RGB"
    public var inputCCTFDecoding: Bool = false
    public var outputColourSpace: String = "sRGB"
    public var outputCCTFEncoding: Bool = true
    public var inputGamutCompress = InputGamutCompressSpec()
    public var outputGamutCompress = OutputGamutCompressSpec()
    public var crop: Bool = false
    public var cropCenter: (x: Double, y: Double) = (0.5, 0.5)
    public var cropSize: (width: Double, height: Double) = (0.1, 0.1)
    public var upscaleFactor: Double = 1.0
    /// Scan the negative instead of printing it first.
    public var scanFilm: Bool = false

    public init() {}

    public static func == (a: IOParams, b: IOParams) -> Bool {
        a.inputColourSpace == b.inputColourSpace && a.inputCCTFDecoding == b.inputCCTFDecoding
            && a.outputColourSpace == b.outputColourSpace
            && a.outputCCTFEncoding == b.outputCCTFEncoding
            && a.inputGamutCompress == b.inputGamutCompress
            && a.outputGamutCompress == b.outputGamutCompress && a.crop == b.crop
            && a.cropCenter == b.cropCenter && a.cropSize == b.cropSize
            && a.upscaleFactor == b.upscaleFactor && a.scanFilm == b.scanFilm
    }
}

/// `DebugParams`.
public struct DebugParams: Sendable, Equatable {
    public var deactivateSpatialEffects: Bool = false
    public var deactivateStochasticEffects: Bool = false
    public var printTimings: Bool = false
    /// Forces the pipeline to a deterministic per-pixel transform, suitable for LUT sampling.
    /// Spatial effects, stochastic effects, auto-exposure and the scanner white, black and unsharp
    /// corrections all turn off regardless of their own settings.
    public var lutMode: Bool = false

    public init() {}
}

/// `TapsParams`. Names the entry and exit points in the pipeline topology. `nil` on both runs
/// end to end, from `rgbIn` to `rgbOut`.
public struct TapsParams: Sendable, Equatable {
    public var inject: Tap?
    public var collect: Tap?

    public init(inject: Tap? = nil, collect: Tap? = nil) {
        self.inject = inject
        self.collect = collect
    }
}

/// `SettingsParams`. Numerical strategy, not look.
public struct SettingsParams: Sendable, Equatable {
    public enum RGBToRawMethod: String, Sendable, CaseIterable {
        /// Works across the full visible locus. The default.
        case hanatos2025
        /// sRGB only, and clips the input.
        case mallett2019
    }

    public var rgbToRawMethod: RGBToRawMethod = .hanatos2025
    public var applyHanatos2025AdaptationWindow: Bool = true
    public var applyHanatos2025AdaptationSurface: Bool = false
    public var spectralGaussianBlur: Double = 0.0
    public var useEnlargerLUT: Bool = false
    public var useScannerLUT: Bool = false
    public var lutResolution: Int = 17
    public var useFastStats: Bool = false
    public var previewMaxSize: Int = 640
    public var previewMode: Bool = false
    public var neutralPrintFiltersFromDatabase: Bool = true

    public init() {}
}

/// `RuntimePhotoParams`. Everything one render needs.
public struct RuntimePhotoParams: Sendable, Equatable {
    public var film: Profile
    public var print: Profile
    public var filmRender = FilmRenderingParams()
    public var printRender = PrintRenderingParams()
    public var camera = CameraParams()
    public var enlarger = EnlargerParams()
    public var scanner = ScannerParams()
    public var io = IOParams()
    public var debug = DebugParams()
    public var settings = SettingsParams()
    public var taps = TapsParams()

    public init(film: Profile, print: Profile) {
        self.film = film
        self.print = print
    }

    /// Resolved input colour space.
    public func inputColourSpace() throws -> ColourSpace {
        try ColourSpace.named(io.inputColourSpace)
    }

    /// Resolved output colour space.
    public func outputColourSpace() throws -> ColourSpace {
        try ColourSpace.named(io.outputColourSpace)
    }

    /// Rejects settings that cannot be satisfied, before any pixels move.
    public func validate() throws {
        _ = try inputColourSpace()
        _ = try outputColourSpace()
        _ = try Illuminant(label: enlarger.illuminant)
        _ = try Illuminant(label: film.info.referenceIlluminant)
        _ = try Illuminant(label: film.info.viewingIlluminant)
        _ = try Illuminant(label: print.info.viewingIlluminant)
        try io.inputGamutCompress.validate()

        // The coarse 3D enlarger and scanner LUTs are not ported. They replace the per-pixel
        // spectral map with an interpolated cube, which the reference calls an approximation, and
        // they default off. Reject them so the flag is never accepted and then ignored.
        if settings.useEnlargerLUT {
            throw SpektraError.unsupportedSetting("settings.use_enlarger_lut", value: "true")
        }
        if settings.useScannerLUT {
            throw SpektraError.unsupportedSetting("settings.use_scanner_lut", value: "true")
        }
        guard film.info.stage == .filming else {
            throw SpektraError.invalidProfile(
                film.info.stock, reason: "cannot act as the negative, stage is \(film.info.stage)")
        }
        guard print.info.stage == .printing else {
            throw SpektraError.invalidProfile(
                print.info.stock,
                reason: "cannot act as the print medium, stage is \(print.info.stage)")
        }
    }
}

extension RuntimePhotoParams {
    /// `init_params(film_profile:, print_profile:)`.
    ///
    /// Returns undigested params, as upstream does. ``Simulator`` runs ``ParamsBuilder/digest`` on
    /// construction, so callers edit the plain values and digest derives the rest.
    public static func make(film: String, print: String) throws -> RuntimePhotoParams {
        let params = RuntimePhotoParams(
            film: try ProfileLibrary.load(film),
            print: try ProfileLibrary.load(print)
        )
        try params.validate()
        return params
    }
}
