import Foundation

/// Black and white point corrections, the way a scanner's auto-levels would apply them.
///
/// Ports `runtime/services/color_reference.py`. Both corrections default off, so every method here
/// returns the identity on the default render path. When either is on, the service maps the measured
/// black and white references onto the requested levels with a clipped linear ramp, and reports the
/// exposure adjustment that keeps 18% grey where it was.
///
/// The scanning stage injects ``cmyToLogXYZ`` during pipeline construction, and the printing stage
/// injects the two print references. Reading a correction before they are set is a programming error,
/// and throws instead of silently correcting against zero.
public final class ColorReferenceService {
    private let film: Profile
    private let print: Profile
    private let scanFilm: Bool
    private let blackCorrection: Bool
    private let whiteCorrection: Bool
    private let blackLevel: Double
    private let whiteLevel: Double

    /// Converts CMY density to log10 XYZ. Set by the scanning stage, since only it knows which
    /// medium and viewing illuminant apply.
    public var cmyToLogXYZ: (@Sendable (ImageBuffer) -> ImageBuffer)?
    /// Log exposure the paper receives through the darkest part of the negative. Set by the printing
    /// stage.
    public var logRawPrintBlack: ImageBuffer?
    /// The same for the lightest part.
    public var logRawPrintWhite: ImageBuffer?

    private var yBlack: Double?
    private var yWhite: Double?

    public init(
        film: Profile,
        print: Profile,
        scanner: ScannerParams,
        io: IOParams
    ) throws {
        self.film = film
        self.print = print
        scanFilm = io.scanFilm
        blackCorrection = scanner.blackCorrection
        whiteCorrection = scanner.whiteCorrection
        blackLevel = try Self.removeSRGBTransfer(scanner.blackLevel)
        whiteLevel = try Self.removeSRGBTransfer(scanner.whiteLevel)
    }

    /// True when neither correction is on, which is the default.
    public var isIdentity: Bool { !blackCorrection && !whiteCorrection }

    // MARK: - Exposure corrections

    /// Applied in the filming stage. Only positive film being scanned directly needs it.
    public func filmingExposureCorrection() throws -> Double {
        if isIdentity { return 1.0 }
        if film.isNegative { return 1.0 }
        guard scanFilm else { return 1.0 }

        try updateReferences(inPrint: false)
        let midgrayDensity = -Foundation.log10(0.184)
        let corrected = try correction().midgrayCorrected
        let correctedDensity = -Foundation.log10(corrected)

        let averageCurve = channelMean(film.data.densityCurves, count: film.data.exposureCount)
        let averageMin = nanMean(film.data.baseDensity)
        let axis = film.data.logExposure

        // The reference negates both arguments so the axis ascends for a positive stock, whose
        // density falls with exposure.
        let negatedCurve = averageCurve.map { -$0 }
        let atCorrected = -Interpolation.npInterp(
            query: [-(correctedDensity - averageMin)], xp: negatedCurve, fp: axis)[0]
        let atMidgray = -Interpolation.npInterp(
            query: [-(midgrayDensity - averageMin)], xp: negatedCurve, fp: axis)[0]
        return 1.0 / Foundation.pow(10.0, atCorrected - atMidgray)
    }

    /// Applied in the printing stage.
    public func printingExposureCorrection() throws -> Double {
        if isIdentity { return 1.0 }
        guard print.isNegative else { return 1.0 }

        try updateReferences(inPrint: true)
        let midgrayDensity = -Foundation.log10(0.184)
        let corrected = try correction().midgrayCorrected
        let correctedDensity = -Foundation.log10(corrected)

        let averageCurve = channelMean(print.data.densityCurves, count: print.data.exposureCount)
        let averageMin = nanMean(print.data.baseDensity)
        let axis = print.data.logExposure

        let atCorrected = Interpolation.npInterp(
            query: [correctedDensity - averageMin], xp: averageCurve, fp: axis)[0]
        let atMidgray = Interpolation.npInterp(
            query: [midgrayDensity - averageMin], xp: averageCurve, fp: axis)[0]
        return Foundation.pow(10.0, atCorrected - atMidgray)
    }

    /// Applied in the scanning stage, scaling XYZ by the correction of its luminance.
    public func correctXYZ(_ xyz: ImageBuffer) throws -> ImageBuffer {
        if isIdentity { return xyz }
        // Negative film scanned directly is left alone: there is no white to anchor to.
        if scanFilm && film.isNegative { return xyz }

        let (apply, _) = try correction()
        var out = xyz
        out.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in stride(from: 0, to: buf.count, by: 3) {
                let y = p[i + 1]
                let scale = apply(y) / (y + 1e-10)
                p[i] *= scale
                p[i + 1] *= scale
                p[i + 2] *= scale
            }
        }
        return out
    }

    // MARK: - References

    private func updateReferences(inPrint: Bool) throws {
        if isIdentity { return }
        if scanFilm && film.isNegative { return }

        guard let cmyToLogXYZ else {
            throw SpektraError.unsupportedSetting(
                "scanner black/white correction",
                value: "cmyToLogXYZ was not configured before use")
        }

        if scanFilm && film.isPositive && !inPrint {
            // Maximum density is black on a positive, zero density is white.
            let maxima = film.data.densityCurveMaxima
            let black = ImageBuffer(height: 1, width: 1, channels: 3, values: maxima)
            let white = ImageBuffer(height: 1, width: 1, channels: 3, repeating: 0)
            yBlack = Foundation.pow(10.0, cmyToLogXYZ(black).values[1])
            yWhite = Foundation.pow(10.0, cmyToLogXYZ(white).values[1])
        } else if !scanFilm && print.isNegative && inPrint {
            guard let logRawPrintBlack, let logRawPrintWhite else {
                throw SpektraError.unsupportedSetting(
                    "scanner black/white correction",
                    value: "the print references were not configured before use")
            }
            let black = DensityCurves.densityFromLogExposure(
                logExposure: logRawPrintBlack, curves: print.data.densityCurves,
                axis: print.data.logExposure, gammaFactor: 1.0)
            let white = DensityCurves.densityFromLogExposure(
                logExposure: logRawPrintWhite, curves: print.data.densityCurves,
                axis: print.data.logExposure, gammaFactor: 1.0)
            yBlack = Foundation.pow(10.0, cmyToLogXYZ(black).values[1])
            yWhite = Foundation.pow(10.0, cmyToLogXYZ(white).values[1])
        }
    }

    /// The clipped linear ramp, and where it maps 18% grey.
    ///
    /// With only one correction enabled, the other end anchors to the measured reference, which
    /// leaves that end untouched.
    private func correction() throws -> (apply: (Double) -> Double, midgrayCorrected: Double) {
        guard let yBlack, let yWhite else {
            throw SpektraError.unsupportedSetting(
                "scanner black/white correction", value: "references have not been measured")
        }
        var white = whiteLevel
        var black = blackLevel
        if blackCorrection && !whiteCorrection { white = yWhite }
        if whiteCorrection && !blackCorrection { black = yBlack }

        let m = (white - black) / (yWhite - yBlack + 1e-10)
        let q = black - m * yBlack
        return ({ y in min(max(m * y + q, 0), 1) }, (0.184 - q) / m)
    }

    // MARK: - Helpers

    /// `_remove_sRGB_cctf`: decode the sRGB transfer function, pass through the near-identity
    /// same-space matrix, then average the three channels.
    static func removeSRGBTransfer(_ level: Double) throws -> Double {
        let sRGB = try ColourSpace.named("sRGB")
        var buffer = ImageBuffer(height: 1, width: 1, channels: 3, repeating: level)
        Colour.RGBToRGB(&buffer, from: sRGB, to: sRGB, applyDecoding: true)
        return buffer.values.reduce(0, +) / 3.0
    }

    /// `np.nanmean(curves, axis=1)`, the mean across channels at each exposure.
    private func channelMean(_ curves: [Double], count: Int) -> [Double] {
        (0..<count).map { i in
            var sum = 0.0
            var n = 0
            for c in 0..<3 {
                let v = curves[i * 3 + c]
                if v.isNaN { continue }
                sum += v
                n += 1
            }
            return n > 0 ? sum / Double(n) : .nan
        }
    }

    private func nanMean(_ values: [Double]) -> Double {
        var sum = 0.0
        var n = 0
        for v in values where !v.isNaN {
            sum += v
            n += 1
        }
        return n > 0 ? sum / Double(n) : .nan
    }
}
