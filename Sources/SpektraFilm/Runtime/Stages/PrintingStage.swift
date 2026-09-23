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

        let bandRows = ImageBuffer.bandRows(
            width: cmyFilmDensity.width, channels: ColourTables.wavelengthCount)
        let logRawPrint = cmyFilmDensity.mapPerPixel(bandRows: bandRows, channelsOut: 3) { band in
            filmCMYToPrintLogRaw(band)
        }

        var raw = logRawPrint
        let exposureScale =
            enlargerParams.printExposure * (try colourReference.printingExposureCorrection())
        raw.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] = Foundation.pow(10.0, p[i]) * exposureScale }
        }

        if enlargerParams.diffusionFilter.active, let pixelSize = resizing.pixelSizeMicrons {
            raw = try Diffusion.applyDiffusionFilter(
                raw, enlargerParams.diffusionFilter, pixelSizeMicrons: pixelSize)
        }

        raw.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] = log10Guard(p[i]) }
        }
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
        let spectral = DensityCurves.spectralDensity(
            cmy: cmyFilmDensity,
            channelDensity: film.data.channelDensity,
            baseDensity: film.data.baseDensity)
        let printIlluminant = enlarger.filteredIlluminant(lampSpectrum)
        let light = DensityCurves.densityToLight(spectral, illuminant: printIlluminant)
        var raw = DensityCurves.project(light, onto: paperSensitivity)

        let midgrayFactor = exposureFactorMidgray(printIlluminant: printIlluminant)
        let preflash = rawPreflash(printIlluminant: printIlluminant)
        raw.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in stride(from: 0, to: buf.count, by: 3) {
                for c in 0..<3 { p[i + c] = p[i + c] * midgrayFactor[c] + preflash[c] }
            }
        }

        raw.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] = log10Guard(p[i]) }
        }
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
