import Foundation

/// Filming: scene light exposed onto the negative, and developed.
///
/// Ports `runtime/stages/filming.py`. Input RGB is reconstructed into a spectrum, integrated against
/// the stock's spectral sensitivities to get camera raw, propagated through the optical effects that
/// happen before and inside the emulsion, then developed into dye density.
public final class FilmingStage {
    private let film: Profile
    private let filmRender: FilmRenderingParams
    private let camera: CameraParams
    private let io: IOParams
    private let settings: SettingsParams
    private let resizing: ResizingService
    private let enlarger: EnlargerService
    private let colourReference: ColorReferenceService
    private let spatial: any SpatialFilter

    private let inputColourSpace: ColourSpace
    private let referenceIlluminant: Illuminant
    /// `10 ** log_sensitivity`, with NaN zeroed, and the band-pass filter folded in when engaged.
    private let sensitivity: [Double]
    private let tcLUT: ImageBuffer?

    /// The 18% grey references the print balance needs. Computed here because they depend on the
    /// film profile and the camera exposure, and consumed by the printing stage.
    public private(set) var densitySpectralMidgray: ImageBuffer?
    public private(set) var densitySpectralMidgrayCompensated: ImageBuffer?

    public init(
        film: Profile,
        filmRender: FilmRenderingParams,
        camera: CameraParams,
        io: IOParams,
        settings: SettingsParams,
        resizing: ResizingService,
        enlarger: EnlargerService,
        colourReference: ColorReferenceService,
        spatial: any SpatialFilter
    ) throws {
        self.film = film
        self.filmRender = filmRender
        self.camera = camera
        self.io = io
        self.settings = settings
        self.resizing = resizing
        self.enlarger = enlarger
        self.colourReference = colourReference
        self.spatial = spatial

        inputColourSpace = try ColourSpace.named(io.inputColourSpace)
        referenceIlluminant = try Illuminant(label: film.info.referenceIlluminant)
        sensitivity = try Self.filmSensitivity(film: film, camera: camera)

        var adaptation = try film.hanatos2025Adaptation()
        adaptation.applyWindow = settings.applyHanatos2025AdaptationWindow
        adaptation.applySurface = settings.applyHanatos2025AdaptationSurface
        adaptation.spectralGaussianBlur = settings.spectralGaussianBlur

        tcLUT = try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: SpectralMatrix(sensitivity),
            adaptation: adaptation,
            gamutCompress: io.inputGamutCompress,
            compressionBake: TCLUTCompressionBake())
    }

    /// `auto_exposure`. Meters a small preview and scales the whole frame.
    public func autoExposure(_ image: ImageBuffer) throws -> ImageBuffer {
        guard camera.autoExposure else { return image }
        let preview = try resizing.smallPreview(image)
        let ev = AutoExposure.measureEV(
            preview,
            colourSpace: inputColourSpace,
            applyCCTFDecoding: io.inputCCTFDecoding,
            method: camera.autoExposureMethod)
        let gain = Foundation.pow(2.0, ev)
        var out = image
        out.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] *= gain }
        }
        return out
    }

    /// Computes the 18% grey references. Must run before the printing stage exposes anything.
    public func prepareMidgrayReferences() throws {
        let midgray = ImageBuffer(height: 1, width: 1, channels: 3, repeating: 0.184)
        densitySpectralMidgray = try simpleRGBToSpectralDensity(midgray)

        if enlarger.printExposureCompensation {
            let scale = Foundation.pow(2.0, camera.exposureCompensationEV)
            let compensated = ImageBuffer(
                height: 1, width: 1, channels: 3, repeating: 0.184 * scale)
            densitySpectralMidgrayCompensated = try simpleRGBToSpectralDensity(compensated)
        } else {
            densitySpectralMidgrayCompensated = nil
        }
    }

    /// `expose`.
    public func expose(_ image: ImageBuffer) throws -> ImageBuffer {
        var raw = try rgbToFilmRaw(image)

        let exposureGain = Foundation.pow(2.0, camera.exposureCompensationEV)
        raw.transformInPlace { $0 * exposureGain }

        let halation = filmRender.halation
        Diffusion.boostHighlights(
            &raw, boostEV: halation.boostEV, boostRange: halation.boostRange,
            protectEV: halation.protectEV)

        if let pixelSize = resizing.pixelSizeMicrons {
            if camera.diffusionFilter.active {
                raw = try Diffusion.applyDiffusionFilter(
                    raw, camera.diffusionFilter, pixelSizeMicrons: pixelSize)
            }
            if camera.lensBlurMicrons > 0 {
                raw = Diffusion.applyGaussianBlur(
                    raw, sigmaMicrons: camera.lensBlurMicrons, pixelSizeMicrons: pixelSize)
            }
            if halation.active {
                raw = Diffusion.applyHalation(
                    raw, halation, pixelSizeMicrons: pixelSize)
            }
        }

        let correction = try colourReference.filmingExposureCorrection()
        raw.transformInPlace { log10Guard($0 * correction) }
        return raw
    }

    /// `develop`.
    public func develop(_ logRaw: ImageBuffer, grainSeed: UInt64 = 0) -> ImageBuffer {
        Develop.film(
            logRaw: logRaw,
            pixelSizeMicrons: resizing.pixelSizeMicrons,
            profile: film,
            dirCouplers: filmRender.dirCouplers,
            grain: filmRender.grain,
            gammaFactor: filmRender.densityCurveGamma,
            grainSeed: grainSeed,
            spatial: spatial
        )
    }

    // MARK: - Spectral reconstruction

    /// `_rgb_to_film_raw`.
    ///
    /// The colour space defaults to the configured input space, which is what `expose` wants. The
    /// midgray reference overrides it; see ``simpleRGBToSpectralDensity(_:)``.
    private func rgbToFilmRaw(
        _ rgb: ImageBuffer,
        colourSpace: ColourSpace? = nil,
        applyCCTFDecoding: Bool? = nil
    ) throws -> ImageBuffer {
        try SpectralUpsampling.rgbToRaw(
            method: settings.rgbToRawMethod,
            rgb: rgb,
            sensitivity: SpectralMatrix(sensitivity),
            colourSpace: colourSpace ?? inputColourSpace,
            applyCCTFDecoding: applyCCTFDecoding ?? io.inputCCTFDecoding,
            referenceIlluminant: referenceIlluminant,
            tcLUT: tcLUT)
    }

    /// `_simple_rgb_to_density_spectral`. No couplers and no grain, so the reference stays a
    /// calibration constant.
    ///
    /// Reconstructs in sRGB with no transfer decoding, whatever the input colour space is.
    /// `_rgb_to_film_raw` declares `color_space="sRGB"` as a default argument, `expose` passes the
    /// real input space, and this caller passes nothing. The midgray reference is therefore always
    /// sRGB-relative. 0.184 in ProPhoto RGB is a different colour: using the input space here
    /// shifts the print exposure factor by 3.2e-5 in log space, a uniform 1.8e-3 error on the
    /// rendered output.
    private func simpleRGBToSpectralDensity(_ rgb: ImageBuffer) throws -> ImageBuffer {
        var raw = try rgbToFilmRaw(rgb, colourSpace: ColourSpace.sRGB, applyCCTFDecoding: false)
        raw.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            // The reference writes log10(raw + 1e-10) here, with no fmax floor.
            for i in 0..<buf.count { p[i] = Foundation.log10(p[i] + 1e-10) }
        }
        let cmy = Develop.simple(
            logRaw: raw,
            logExposure: film.data.logExposure,
            curves: film.data.densityCurves,
            gammaFactor: filmRender.densityCurveGamma)
        return DensityCurves.spectralDensity(
            cmy: cmy,
            channelDensity: film.data.channelDensity,
            baseDensity: film.data.baseDensity)
    }

    /// Sensitivities in linear units, with the camera's UV and IR cut folded in when either is
    /// engaged. Both amplitudes default to 0, so the default path skips this.
    static func filmSensitivity(film: Profile, camera: CameraParams) throws -> [Double] {
        var sensitivity = nanToNum(
            film.data.logSensitivity.map { Foundation.pow(10.0, $0) })
        guard camera.filterUV.amplitude > 0 || camera.filterIR.amplitude > 0 else {
            return sensitivity
        }

        let illuminant = try Illuminant(label: film.info.referenceIlluminant).spectrum
        let bandPass = Erf.bandPassFilter(
            uv: (camera.filterUV.amplitude, camera.filterUV.wavelength, camera.filterUV.width),
            ir: (camera.filterIR.amplitude, camera.filterIR.wavelength, camera.filterIR.width))

        // Renormalise per channel so the filter reshapes the sensitivity without changing its
        // total response to the reference illuminant.
        var filtered = [Double](repeating: 0, count: 3)
        var unfiltered = [Double](repeating: 0, count: 3)
        for l in 0..<ColourTables.wavelengthCount {
            for c in 0..<3 {
                let s = sensitivity[l * 3 + c] * illuminant[l]
                filtered[c] += s * bandPass[l]
                unfiltered[c] += s
            }
        }
        for l in 0..<ColourTables.wavelengthCount {
            for c in 0..<3 {
                sensitivity[l * 3 + c] *= bandPass[l] / (filtered[c] / unfiltered[c])
            }
        }
        return sensitivity
    }
}
