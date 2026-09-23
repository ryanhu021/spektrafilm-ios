import Testing

@testable import SpektraFilm

/// Checks the resampler against skimage.
///
/// The resampler is on the default render path, though no parameter mentions it: auto-exposure is
/// on by default and meters on a 256 px preview, so every render of a larger photo uses it.
@Suite("Resampling parity")
struct ResampleParityTests {

    static let sizes = ["80x53", "odd_53x97", "square_64"]
    static let factors: [(Double, String)] = [(0.4, "0p4"), (0.213, "0p213"), (0.5, "0p5"), (0.77, "0p77")]

    @Test("order 0 with anti-aliasing", arguments: sizes, factors)
    func orderZero(size: String, factor: (Double, String)) throws {
        let input = try Golden("resample_input_\(size)").imageBuffer()
        let out = try SkimageResampler().rescale(input, factor: factor.0, order: 0)
        let expected = try Golden("resample_order0_\(size)_\(factor.1)")
        #expect(
            [out.height, out.width, out.channels] == expected.shape,
            "shape \(out.height)x\(out.width)x\(out.channels) against \(expected.shape)")
        try expectParity(out.values, matches: "resample_order0_\(size)_\(factor.1)")
    }

    /// The whole metering path: downsample, then measure. The EV reaches the render as a gain, so
    /// an error here scales every pixel.
    @Test("the metered EV after the preview downsample")
    func meteredEV() throws {
        let preview = try Golden("resample_preview_input").imageBuffer()
        let small = try SkimageResampler().rescale(preview, factor: 64.0 / 160.0, order: 0)
        try expectParity(small.values, matches: "resample_preview_256")

        let ev = AutoExposure.measureEV(
            small,
            colourSpace: try ColourSpace.named("Display P3"),
            applyCCTFDecoding: false,
            method: .centerWeighted)
        try expectParity([ev], matches: "resample_preview_ev")
    }

    /// Anti-aliasing is easy to omit, so check that it runs: a nearest pick with no prefilter would
    /// reproduce input samples exactly.
    @Test("the nearest-neighbour path really is prefiltered")
    func antiAliasingRuns() throws {
        let input = try Golden("resample_input_square_64").imageBuffer()
        let filtered = try SkimageResampler().rescale(input, factor: 0.25, order: 0)
        let bare = SkimageResampler.zoomNearest(input, outHeight: 16, outWidth: 16)
        let report = parity(filtered.values, bare.values)
        #expect(
            report.maxAbsolute > 0.01,
            "prefiltered and bare nearest agree to \(report.maxAbsolute), so AA is not running")
    }

    /// Order 3 is reachable only from io.upscaleFactor, and is not implemented.
    @Test("order 3 is refused rather than approximated")
    func orderThreeRefused() throws {
        let input = try Golden("resample_input_square_64").imageBuffer()
        #expect(throws: (any Error).self) {
            try SkimageResampler().rescale(input, factor: 2.0, order: 3)
        }
    }
    /// Evaluating only the kept samples must give the same bits as filtering the whole frame and
    /// then sampling it, on odd sizes and uneven factors in each axis.
    @Test(
        "sampling only the kept pixels equals filtering the whole frame",
        arguments: [(97, 131, 0.3), (240, 64, 0.11), (50, 400, 0.5)])
    func sparseEqualsFull(height: Int, width: Int, factor: Double) throws {
        var values = [Double](repeating: 0, count: height * width * 3)
        for i in values.indices { values[i] = Double((i * 7919) % 1000) / 1000.0 }
        let image = ImageBuffer(height: height, width: width, channels: 3, values: values)
        let sparse = try SkimageResampler().rescale(image, factor: factor, order: 0)

        let outHeight = Int((Double(height) * factor).rounded(.toNearestOrEven))
        let outWidth = Int((Double(width) * factor).rounded(.toNearestOrEven))
        var full = image
        let sigmaY = max(0, (Double(height) / Double(outHeight) - 1) / 2)
        let sigmaX = max(0, (Double(width) / Double(outWidth) - 1) / 2)
        if sigmaY > 1e-15 { full = SkimageResampler.gaussianRows(full, sigma: sigmaY) }
        if sigmaX > 1e-15 { full = SkimageResampler.gaussianColumns(full, sigma: sigmaX) }
        full = SkimageResampler.zoomNearest(full, outHeight: outHeight, outWidth: outWidth)

        #expect(sparse.height == full.height && sparse.width == full.width)
        let differing = zip(sparse.values, full.values).filter { $0.bitPattern != $1.bitPattern }
        #expect(differing.isEmpty, "\(differing.count) values differ")
    }
}
