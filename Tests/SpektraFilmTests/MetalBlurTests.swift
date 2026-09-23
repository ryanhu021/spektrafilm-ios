#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// The Metal blurs against ``GaussianFilter`` and ``ExponentialFilter``.
@Suite("Metal blur", .enabled(if: MetalContext.shared != nil))
struct MetalBlurTests {
    /// Odd, non-square, and noisy, so a transposed pass or a wrong boundary shows.
    static let image: ImageBuffer = {
        let h = 97
        let w = 131
        var v = [Double](repeating: 0, count: h * w * 3)
        for i in v.indices { v[i] = Double((i * 7919) % 1000) / 1000.0 }
        return ImageBuffer(height: h, width: w, channels: 3, values: v)
    }()

    @Test(
        "the Gaussian matches on both paths",
        arguments: [
            [0.0, 0.03, 0.65], [1.5, 2.99, 3.0], [5.0, 20.0, 65.0],
        ])
    func gaussian(sigmas: [Double]) throws {
        let c = try #require(MetalContext.shared)
        let cpu = GaussianFilter.apply(Self.image, sigmaPerChannel: sigmas)
        let gpu = try MetalBlur.gaussian(
            c, try GPUFrame(c, uploading: Self.image), sigmaPerChannel: sigmas
        ).download()
        let worst = zip(cpu.values, gpu.values).map { abs($0 - $1) }.max()!
        // Measured at 5e-8 to 1.5e-7. Without the IIR's double-float state, sigma 65 was 3.3e-3.
        #expect(worst < 1e-5, "sigmas \(sigmas): worst difference \(worst)")
    }

    @Test("the exponential mixture matches")
    func exponential() throws {
        let c = try #require(MetalContext.shared)
        let decays = [0.8, 2.5, 12.0]
        let cpu = ExponentialFilter.apply(Self.image, decayPerChannel: decays)
        let gpu = try MetalBlur.exponential(
            c, try GPUFrame(c, uploading: Self.image), decayPerChannel: decays
        ).download()
        let worst = zip(cpu.values, gpu.values).map { abs($0 - $1) }.max()!
        // Measured at 1.0e-7.
        #expect(worst < 1e-5, "worst difference \(worst)")
    }
}
#endif
