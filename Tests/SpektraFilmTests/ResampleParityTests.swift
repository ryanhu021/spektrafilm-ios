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
}
