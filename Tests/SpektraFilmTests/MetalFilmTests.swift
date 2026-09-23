#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// The spectral upsampling and density-curve kernels against the CPU operators.
@Suite("Metal film kernels", .enabled(if: MetalContext.shared != nil))
struct MetalFilmTests {

    static func converter(film: String, space: String) throws -> Hanatos2025RawConverter {
        let profile = try ProfileLibrary.load(film)
        let sensitivity = nanToNum(profile.data.logSensitivity.map { Foundation.pow(10.0, $0) })
        let lut = try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: SpectralMatrix(sensitivity), adaptation: try profile.hanatos2025Adaptation(),
            gamutCompress: InputGamutCompressSpec(), compressionBake: TCLUTCompressionBake())
        return Hanatos2025RawConverter(
            colourSpace: try ColourSpace.named(space), applyCCTFDecoding: space == "sRGB",
            referenceIlluminant: try Illuminant(label: profile.info.referenceIlluminant), tcLUT: lut)
    }

    /// Every caller takes log10 of the raw next, so the bound is on that.
    @Test(
        "RGB to raw matches the CPU",
        arguments: [("kodak_portra_400", "ProPhoto RGB"), ("fujifilm_velvia_100", "sRGB")])
    func rgbToRaw(film: String, space: String) throws {
        let c = try #require(MetalContext.shared)
        let converter = try Self.converter(film: film, space: space)
        let lut = try #require(converter.tcLUT)
        let input = try Golden("photo_input").imageBuffer()
        let cpu = converter.raw(rgb: input)
        let gpu = try MetalFilm.rgbToRaw(
            c, try GPUFrame(c, uploading: input), converter: converter,
            lut: try c.buffer(from: lut.values), lutSize: lut.height
        ).download()
        let worst = zip(cpu.values, gpu.values).map { abs(log10Guard($0) - log10Guard($1)) }.max()!
        // Measured at 4.2e-7 to 4.4e-7.
        #expect(worst < 2e-6, "worst log10 difference \(worst)")
    }

    @Test(
        "the density-curve lookup matches the CPU",
        arguments: ["kodak_portra_400", "fujifilm_velvia_100", "fujifilm_c200"])
    func densityLookup(film: String) throws {
        let c = try #require(MetalContext.shared)
        let profile = try ProfileLibrary.load(film)
        let data = profile.data
        let curves = DensityCurves.normalized(
            curves: data.densityCurves, minima: data.densityCurveMinima)
        // Log exposures spanning the whole axis and past both ends, plus a NaN.
        var values: [Double] = [.nan, -10, 10]
        for i in 0..<3003 { values.append(-4 + Double(i) * 8 / 3003) }
        let input = ImageBuffer(height: 1, width: values.count / 3, channels: 3, values: values)
        for gamma in [(1.0, 1.0, 1.0), (0.8, 1.1, 1.3)] {
            let cpu = DensityCurves.densityFromLogExposure(
                logExposure: input, curves: curves, axis: data.logExposure, gammaFactor: gamma)
            let table = try MetalFilm.CurveTable(
                c, curves: curves, logExposure: data.logExposure, gamma: gamma)
            let frame = try GPUFrame(c, uploading: input)
            try MetalFilm.interpolate(c, frame, into: frame, table: table)
            let gpu = frame.download()
            var worst = 0.0
            for (a, b) in zip(cpu.values, gpu.values) where !(a.isNaN && b.isNaN) {
                worst = max(worst, a.isNaN != b.isNaN ? .infinity : abs(a - b))
            }
            // Measured at 2.2e-7 to 4.6e-7.
            #expect(worst < 1e-5, "\(film) gamma \(gamma): worst density difference \(worst)")
        }
    }
}
#endif
