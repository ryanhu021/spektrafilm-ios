import Foundation

/// Printing: the developed negative projected onto paper, and developed again.
///
/// Ports `runtime/stages/printing.py`. The negative's dye densities become a transmitted spectrum,
/// the enlarger's filtered lamp shines through it, and the paper's spectral sensitivities integrate
/// that into an exposure. Paper needs no coupler model: it never samples a scene, so it is designed
/// with little channel cross-talk to begin with.
public final class PrintingStage {
    private let film: Profile
    private let filmRender: FilmRenderingParams
    private let print: Profile
    private let printRender: PrintRenderingParams
    private let enlargerParams: EnlargerParams
    private let settings: SettingsParams
    private let enlarger: EnlargerService
    private let resizing: ResizingService
    private let colourReference: ColorReferenceService
    private let spatial: any SpatialFilter

    private let paperSensitivity: [Double]
    private let lampSpectrum: [Double]

    public init(
        film: Profile,
        filmRender: FilmRenderingParams,
        print: Profile,
        printRender: PrintRenderingParams,
        enlargerParams: EnlargerParams,
        settings: SettingsParams,
        enlarger: EnlargerService,
        resizing: ResizingService,
        colourReference: ColorReferenceService,
        spatial: any SpatialFilter
    ) throws {
        self.film = film
        self.filmRender = filmRender
        self.print = print
        self.printRender = printRender
        self.enlargerParams = enlargerParams
        self.settings = settings
        self.enlarger = enlarger
        self.resizing = resizing
        self.colourReference = colourReference
        self.spatial = spatial

        paperSensitivity = nanToNum(
            print.data.logSensitivity.map { Foundation.pow(10.0, $0) })
        lampSpectrum = try Illuminant(label: enlargerParams.illuminant).spectrum
    }

    /// `expose`.
    public func expose(_ cmyFilmDensity: ImageBuffer) throws -> ImageBuffer {
        // The colour reference service needs the paper exposure at the negative's extremes. Grain's
        // densityMin sets the floor; the curves' per-channel maxima set the ceiling.
        let black = ImageBuffer(
            height: 1, width: 1, channels: 3,
            values: [
                -filmRender.grain.densityMin.0,
                -filmRender.grain.densityMin.1,
                -filmRender.grain.densityMin.2,
            ])
        let white = ImageBuffer(
            height: 1, width: 1, channels: 3, values: film.data.densityCurveMaxima)
        colourReference.logRawPrintBlack = filmCMYToPrintLogRaw(black)
        colourReference.logRawPrintWhite = filmCMYToPrintLogRaw(white)

        // One binding for the frame: a second one keeps the storage shared, and the in-place passes
        // below then copy it.
        var raw = filmCMYToPrintLogRaw(cmyFilmDensity)

        let exposureScale =
            enlargerParams.printExposure * (try colourReference.printingExposureCorrection())
        raw.transformInPlace { Foundation.pow(10.0, $0) * exposureScale }

        if enlargerParams.diffusionFilter.active, let pixelSize = resizing.pixelSizeMicrons {
            raw = try Diffusion.applyDiffusionFilter(
                raw, enlargerParams.diffusionFilter, pixelSizeMicrons: pixelSize)
        }

        raw.transformInPlace { log10Guard($0) }
        return raw
    }

    /// `develop`.
    public func develop(_ logRaw: ImageBuffer) throws -> ImageBuffer {
        try Develop.print(
            logRaw: logRaw, profile: print, morph: printRender.densityCurvesMorph)
    }

    // MARK: - The spectral map

    /// `_film_cmy_to_print_log_raw`.
    private func filmCMYToPrintLogRaw(_ cmyFilmDensity: ImageBuffer) -> ImageBuffer {
        let printIlluminant = enlarger.filteredIlluminant(lampSpectrum)
        var raw = SpectralContraction.project(
            cmy: cmyFilmDensity,
            channelDensity: film.data.channelDensity,
            baseDensity: film.data.baseDensity,
            illuminant: printIlluminant,
            response: paperSensitivity)

        let midgrayFactor = exposureFactorMidgray(printIlluminant: printIlluminant)
        let preflash = rawPreflash(printIlluminant: printIlluminant)
        raw.transformInPlace { channel, value in
            value * midgrayFactor[channel] + preflash[channel]
        }
        raw.transformInPlace { log10Guard($0) }
        return raw
    }

    /// `_compute_raw_preflash`. Pre-flashing exposes the paper through the film base only, which
    /// lifts the shadows and holds highlights.
    private func rawPreflash(printIlluminant: [Double]) -> [Double] {
        guard enlargerParams.preflashExposure > 0 else { return [0, 0, 0] }
        let preflashIlluminant = enlarger.preflashIlluminant(lampSpectrum)
        let base = ImageBuffer(
            height: 1, width: 1, channels: ColourTables.wavelengthCount,
            values: film.data.baseDensity)
        let light = DensityCurves.densityToLight(base, illuminant: preflashIlluminant)
        let raw = DensityCurves.project(light, onto: paperSensitivity)
        return raw.values.map { $0 * enlargerParams.preflashExposure }
    }

    /// `_compute_exposure_factor_midgray`.
    ///
    /// Two independent switches. `normalizePrintExposure` puts an 18% grey patch at the paper's own
    /// midscale; `printExposureCompensation` follows the camera's exposure compensation so changing
    /// the negative exposure does not also change the print's brightness.
    private func exposureFactorMidgray(printIlluminant: [Double]) -> [Double] {
        guard let midgray = enlarger.densitySpectralMidgray else { return [1, 1, 1] }
        let factor = Self.exposureFactor(
            sensitivity: paperSensitivity, illuminant: printIlluminant, midgray: midgray)

        let compensated: [Double]
        if let midgrayCompensated = enlarger.densitySpectralMidgrayCompensated {
            compensated = Self.exposureFactor(
                sensitivity: paperSensitivity, illuminant: printIlluminant,
                midgray: midgrayCompensated)
        } else {
            compensated = [1, 1, 1]
        }

        let compensate = enlargerParams.printExposureCompensation
        let normalize = enlargerParams.normalizePrintExposure
        if compensate && !normalize {
            return (0..<3).map { compensated[$0] / factor[$0] }
        } else if normalize && compensate {
            return compensated
        } else if normalize && !compensate {
            return factor
        }
        return [1, 1, 1]
    }

    /// `_exposure_factor`. The geometric mean across channels normalises the exposure without
    /// changing the colour balance.
    static func exposureFactor(
        sensitivity: [Double], illuminant: [Double], midgray: ImageBuffer
    ) -> [Double] {
        let light = DensityCurves.densityToLight(midgray, illuminant: illuminant)
        let raw = DensityCurves.project(light, onto: sensitivity)
        var logSum = 0.0
        for c in 0..<3 { logSum += Foundation.log(max(raw.values[c], 1e-10)) }
        let geometricMean = Foundation.exp(logSum / 3.0)
        return [Double](repeating: 1.0 / geometricMean, count: 3)
    }
}

/// The spectral map both ``PrintingStage`` and ``ScanningStage`` run over the frame: CMY density to
/// a spectrum, the spectrum lit by an illuminant, that light projected onto a three-column response.
///
/// Composed from `DensityCurves.spectralDensity`, `densityToLight` and `project`, the middle step
/// copies its 81-channel input because the caller still holds it, so two spectral buffers are live
/// at once. Here each pixel's spectrum is formed and contracted in registers, so no spectral buffer
/// exists at all, and the callers no longer band the frame: `mapPerPixel` was there to bound the
/// spectral intermediate, and its row-band copies were the only thing left in the budget.
/// Wavelengths are summed in ascending order, as `project` sums them, so the result is
/// bit-identical.
enum SpectralContraction {
    static func project(
        cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        response: [Double],
        scale: Double = 1.0
    ) -> ImageBuffer {
        let wavelengths = ColourTables.wavelengthCount
        precondition(cmy.channels == 3, "density buffer must have 3 channels")
        precondition(
            channelDensity.count == wavelengths * 3, "channel_density must be \(wavelengths) x 3")
        precondition(baseDensity.count == wavelengths, "base_density must be \(wavelengths)")
        precondition(
            illuminant.count == wavelengths,
            "illuminant has \(illuminant.count) samples, expected \(wavelengths)")
        precondition(response.count == wavelengths * 3, "response must be \(wavelengths) x 3")

        // One row per wavelength: the three dye weights, the base density, the incident light, and
        // the three response columns. Interleaving them keeps the inner loop to a single stride.
        var table = [Double](repeating: 0, count: wavelengths * 8)
        for l in 0..<wavelengths {
            table[l * 8] = channelDensity[l * 3]
            table[l * 8 + 1] = channelDensity[l * 3 + 1]
            table[l * 8 + 2] = channelDensity[l * 3 + 2]
            table[l * 8 + 3] = baseDensity[l]
            table[l * 8 + 4] = illuminant[l]
            table[l * 8 + 5] = response[l * 3]
            table[l * 8 + 6] = response[l * 3 + 1]
            table[l * 8 + 7] = response[l * 3 + 2]
        }

        var out = ImageBuffer(height: cmy.height, width: cmy.width, channels: 3)
        cmy.values.withUnsafeBufferPointer { src in
            table.withUnsafeBufferPointer { weights in
                out.values.withUnsafeMutableBufferPointer { dst in
                    let s = src.baseAddress!
                    let t = weights.baseAddress!
                    let d = dst.baseAddress!
                    for p in 0..<cmy.pixelCount {
                        let c = s[p * 3]
                        let m = s[p * 3 + 1]
                        let y = s[p * 3 + 2]
                        var acc = (0.0, 0.0, 0.0)
                        for l in 0..<wavelengths {
                            let w = l * 8
                            var density = c * t[w] + m * t[w + 1] + y * t[w + 2]
                            density += t[w + 3]
                            let transmitted = Foundation.pow(10.0, -density) * t[w + 4]
                            let light = transmitted.isNaN ? 0 : transmitted
                            acc.0 += light * t[w + 5]
                            acc.1 += light * t[w + 6]
                            acc.2 += light * t[w + 7]
                        }
                        d[p * 3] = acc.0 * scale
                        d[p * 3 + 1] = acc.1 * scale
                        d[p * 3 + 2] = acc.2 * scale
                    }
                }
            }
        }
        return out
    }
}
