#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// The output gamut compression kernel against ``OutputGamutCompressor``.
///
/// The difference is `|gpu - cpu| / max(1, |cpu|)`: absolute on displayable values, relative on
/// the few outputs far outside the cube, where float32 spacing alone exceeds the absolute figure.
/// The CPU runs on the float64 fixture, so the input's own rounding to float32 is included.
@Suite("Metal gamut compression", .enabled(if: MetalContext.shared != nil))
struct MetalGamutTests {

    static let spaces = ["sRGB", "Display P3", "ITU-R BT.2020", "ProPhoto RGB"]

    static let knees: [(name: String, knee: Knee, lightness: Knee?)] = [
        ("default", GamutParityTests.defaultKnee, GamutParityTests.lightnessKnee),
        ("soft", GamutParityTests.softKnee, nil),
        ("late", GamutParityTests.lateKnee, GamutParityTests.lightnessKnee),
    ]

    static let fixtures = ["gamut_rgb_input", "gamut_realizable_input"]

    static func compressor(
        _ algorithm: OutputGamutCompressSpec.Algorithm, _ space: String?, knee: Knee,
        lightness: Knee?
    ) throws -> OutputGamutCompressor {
        var spec = OutputGamutCompressSpec()
        spec.algorithm = algorithm
        spec.knee = knee
        spec.lightnessCompression = lightness
        return try OutputGamutCompressor(
            spec: spec, colourSpace: try space.map { try ColourSpace.named($0) })
    }

    /// The largest difference where both sides are finite, and the number of values where the two
    /// sides disagree on being NaN or on an infinity.
    static func compare(
        _ context: MetalContext, _ compressor: OutputGamutCompressor, _ values: [Double]
    ) throws -> (worst: Double, mismatches: Int) {
        let input = ImageBuffer(height: 1, width: values.count / 3, channels: 3, values: values)
        var cpu = input
        compressor.apply(to: &cpu)
        let frame = try GPUFrame(context, uploading: input)
        try MetalGamut.apply(context, compressor, to: frame)
        let gpu = frame.download()
        var worst = 0.0
        var mismatches = 0
        for (a, b) in zip(cpu.values, gpu.values) {
            if a.isFinite && b.isFinite {
                worst = max(worst, abs(a - b) / max(1, abs(a)))
            } else if !(a.isNaN && b.isNaN) && a != b {
                mismatches += 1
            }
        }
        return (worst, mismatches)
    }

    /// Worst over the two sweeps in four colour spaces with three knee settings, measured at:
    /// oklch 1.1e-5, oklrab 1.0e-5, jzazbz 3.3e-5, cam16ucs 3.6e-5. sRGB alone: 6.1e-6, 5.8e-6,
    /// 2.6e-5 and 7.8e-6.
    ///
    /// JzAzBz's PQ exponent of 134 amplifies each float32 rounding of the cone response, and the
    /// opponent axes then cancel it against its neighbours, so the hue and the `C_max` it indexes
    /// carry the error. CAM16's worst pixel is a near-black ProPhoto blue whose negative cone
    /// response cancels in `A`, and whose output moves 18 times as far as its lightness does.
    @Test(
        "the perceptual algorithms match the CPU",
        arguments: [
            (OutputGamutCompressSpec.Algorithm.oklch, 2e-5), (.oklrab, 2e-5), (.jzazbz, 5e-5),
            (.cam16ucs, 5e-5),
        ])
    func perceptual(algorithm: OutputGamutCompressSpec.Algorithm, gate: Double) throws {
        let c = try #require(MetalContext.shared)
        var worst = 0.0
        for space in Self.spaces {
            for knee in Self.knees {
                let compressor = try Self.compressor(
                    algorithm, space, knee: knee.knee, lightness: knee.lightness)
                for fixture in Self.fixtures {
                    let result = try Self.compare(c, compressor, try Golden(fixture).values)
                    let label = "\(space) \(knee.name) \(fixture)"
                    #expect(result.mismatches == 0, "\(label): \(result.mismatches) mismatches")
                    worst = max(worst, result.worst)
                }
            }
        }
        #expect(worst < gate, "\(algorithm.rawValue): worst difference \(worst)")
    }

    /// Measured at 2.2e-6.
    @Test("aces_rgc matches the CPU")
    func acesRGC() throws {
        let c = try #require(MetalContext.shared)
        var worst = 0.0
        for knee in Self.knees {
            let compressor = try Self.compressor(
                .acesRGC, nil, knee: knee.knee, lightness: knee.lightness)
            for fixture in Self.fixtures {
                let result = try Self.compare(c, compressor, try Golden(fixture).values)
                #expect(
                    result.mismatches == 0, "\(knee.name) \(fixture): \(result.mismatches) mismatches")
                worst = max(worst, result.worst)
            }
        }
        #expect(worst < 5e-6, "worst difference \(worst)")
    }

    /// NaN and infinite channels must come out NaN, or a number, exactly where the CPU's do.
    /// Measured on the finite values at 1.5e-6 for jzazbz and 4.2e-7 at most for the rest.
    @Test(
        "non-finite pixels match the CPU",
        arguments: ["aces_rgc", "oklch", "oklrab", "jzazbz", "cam16ucs"])
    func nonFinite(name: String) throws {
        let c = try #require(MetalContext.shared)
        let algorithm = try #require(OutputGamutCompressSpec.Algorithm(rawValue: name))
        let compressor = try Self.compressor(
            algorithm, name == "aces_rgc" ? nil : "sRGB", knee: GamutParityTests.defaultKnee,
            lightness: GamutParityTests.lightnessKnee)
        let result = try Self.compare(c, compressor, try Golden("gamut_nonfinite_input").values)
        #expect(result.mismatches == 0, "\(result.mismatches) mismatches")
        #expect(result.worst < 3e-6, "worst difference \(result.worst)")
    }

    /// Negative luminance, which the pipeline never produces. CAM16 has no meaning there: `J < 0`
    /// clamps to `eps` in the inverse and the output is ill-conditioned, measured at 7.4e-5
    /// against 7.1e-6 for jzazbz and under 5e-7 for the rest.
    @Test(
        "negative-luminance pixels match the CPU",
        arguments: ["aces_rgc", "oklch", "oklrab", "jzazbz", "cam16ucs"])
    func negativeLuminance(name: String) throws {
        let c = try #require(MetalContext.shared)
        let algorithm = try #require(OutputGamutCompressSpec.Algorithm(rawValue: name))
        let compressor = try Self.compressor(
            algorithm, name == "aces_rgc" ? nil : "sRGB", knee: GamutParityTests.defaultKnee,
            lightness: GamutParityTests.lightnessKnee)
        let result = try Self.compare(c, compressor, try Golden("gamut_negative_input").values)
        #expect(result.mismatches == 0, "\(result.mismatches) mismatches")
        let gate = name == "cam16ucs" ? 2e-4 : 2e-5
        #expect(result.worst < gate, "worst difference \(result.worst)")
    }

    /// Metal's `atan2(0, 0)` is NaN, where C's is 0 and black depends on it.
    @Test("black stays exactly black in every algorithm")
    func black() throws {
        let c = try #require(MetalContext.shared)
        for algorithm in OutputGamutCompressSpec.Algorithm.allCases {
            let space = algorithm == .off || algorithm == .acesRGC ? nil : "sRGB"
            let compressor = try Self.compressor(
                algorithm, space, knee: GamutParityTests.defaultKnee,
                lightness: GamutParityTests.lightnessKnee)
            let frame = try GPUFrame(c, height: 1, width: 1, channels: 3)
            for k in 0..<3 { frame.floats[k] = 0 }
            try MetalGamut.apply(c, compressor, to: frame)
            #expect(frame.download().values == [0, 0, 0], "\(algorithm.rawValue) moved black")
        }
    }

    @Test("off leaves the frame untouched")
    func off() throws {
        let c = try #require(MetalContext.shared)
        let compressor = try Self.compressor(
            .off, nil, knee: GamutParityTests.defaultKnee, lightness: nil)
        let values = try Golden("gamut_rgb_input").values
        let frame = try GPUFrame(
            c, uploading: ImageBuffer(height: 1, width: values.count / 3, channels: 3, values: values))
        let before = frame.download()
        try MetalGamut.apply(c, compressor, to: frame)
        #expect(frame.download().values == before.values)
    }
}
#endif
