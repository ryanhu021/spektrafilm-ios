#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// The elementwise kernels against the CPU expressions they mirror, at float32 precision.
@Suite("Metal elementwise", .enabled(if: MetalContext.shared != nil))
struct MetalElementwiseTests {
    /// Values across the ranges the pipeline feeds these maps, with the edge cases: zero,
    /// negatives, a NaN and values either side of each transfer function's breakpoint.
    static let sample: [Double] = {
        var v: [Double] = [0, -0.5, -1e-6, .nan, 0.0031308, 0.00313, 0.0032, 0.018, 0.081, 1, 2.5]
        for i in 0..<1011 { v.append(Double(i) / 400.0 - 0.1) }
        return v
    }()

    static func frame(_ values: [Double]) throws -> GPUFrame {
        let padded = values + [Double](repeating: 0.25, count: (3 - values.count % 3) % 3)
        let image = ImageBuffer(height: 1, width: padded.count / 3, channels: 3, values: padded)
        return try GPUFrame(MetalContext.shared!, uploading: image)
    }

    /// Worst relative difference, treating two NaNs as equal and a NaN mismatch as infinite.
    static func worst(_ gpu: [Double], _ cpu: [Double]) -> Double {
        var worst = 0.0
        for (a, b) in zip(gpu, cpu) {
            if a.isNaN || b.isNaN {
                if a.isNaN != b.isNaN { return .infinity }
                continue
            }
            worst = max(worst, abs(a - b) / max(1e-6, abs(b)))
        }
        return worst
    }

    @Test("the scalar maps match")
    func scalarMaps() throws {
        let c = try #require(MetalContext.shared)
        let cases: [(String, (GPUFrame) throws -> Void, (Double) -> Double)] = [
            ("scale", { try MetalElementwise.scale(c, $0, by: 1.7) }, { $0 * 1.7 }),
            (
                "log10Guard", { try MetalElementwise.scaleLog10Guard(c, $0, scale: 3.0) },
                { log10Guard($0 * 3.0) }
            ),
            (
                "exp10", { try MetalElementwise.exp10Scale(c, $0, scale: 0.5) },
                { Foundation.pow(10.0, $0) * 0.5 }
            ),
        ]
        for (name, gpu, cpu) in cases {
            let frame = try Self.frame(Self.sample)
            try gpu(frame)
            let got = Array(frame.download().values.prefix(Self.sample.count))
            let expected = Self.sample.map(cpu)
            // log10Guard's output crosses zero, where a relative bound means nothing, and the
            // density lookup downstream sees its absolute error.
            let error =
                name == "log10Guard"
                ? zip(got, expected).map { abs($0 - $1) }.max() ?? 0
                : Self.worst(got, expected)
            #expect(error < 2e-6, "\(name): worst error \(error)")
        }
    }

    @Test("every transfer function matches", arguments: TransferFunction.allCases)
    func transfer(function: TransferFunction) throws {
        let c = try #require(MetalContext.shared)
        for encode in [true, false] {
            let frame = try Self.frame(Self.sample)
            try MetalElementwise.transfer(c, frame, function, encode: encode)
            let got = Array(frame.download().values.prefix(Self.sample.count))
            let expected = Self.sample.map { encode ? function.encode($0) : function.decode($0) }
            let error = Self.worst(got, expected)
            #expect(error < 2e-6, "\(function) \(encode ? "encode" : "decode"): \(error)")
        }
    }

    @Test("the per-pixel maps match")
    func pixelMaps() throws {
        let c = try #require(MetalContext.shared)
        let m = Matrix3(0.9, 0.2, -0.1, 0.05, 1.1, 0.0, -0.02, 0.1, 0.95)
        let base = Array(Self.sample.dropFirst(11))
        let other = base.map { $0 * 0.8 + 0.1 }

        let x = try Self.frame(base)
        try MetalElementwise.matrix3(c, x, m)
        var cpu = ImageBuffer(height: 1, width: base.count / 3, channels: 3, values: base)
        m.apply(to: &cpu)
        #expect(Self.worst(Array(x.download().values.prefix(base.count)), cpu.values) < 2e-6)

        let a = try Self.frame(base)
        try MetalElementwise.affine3(c, a, factor: [1.5, 0.5, 2], offset: [0.1, 0, -0.2])
        let affine = base.enumerated().map { i, v in v * [1.5, 0.5, 2][i % 3] + [0.1, 0, -0.2][i % 3] }
        #expect(Self.worst(Array(a.download().values.prefix(base.count)), affine) < 2e-6)

        for (name, amount) in [("unsharp", 0.7), ("mix", 0.3), ("reverse", 0.0)] {
            let a = try Self.frame(base)
            let b = try Self.frame(other)
            switch name {
            case "unsharp": try MetalElementwise.unsharp(c, a, b, amount: amount)
            case "mix": try MetalElementwise.mix(c, a, b, weight: amount)
            default: try MetalElementwise.reverseSubtract(c, a, b)
            }
            let expected = zip(base, other).map { v, w in
                switch name {
                case "unsharp": return v + amount * (v - w)
                case "mix": return (1 - amount) * v + amount * w
                default: return w - v
                }
            }
            let got = Array(a.download().values.prefix(base.count))
            // Absolute here: these differences cross zero, where a relative bound is meaningless.
            let error = zip(got, expected).map { abs($0 - $1) }.max() ?? 0
            #expect(error < 1e-6, "\(name): worst absolute error \(error)")
        }
    }
}
#endif
