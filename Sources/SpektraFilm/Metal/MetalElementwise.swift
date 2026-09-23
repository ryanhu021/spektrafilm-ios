#if canImport(Metal)
import Foundation
import Metal

/// Wrappers for ``MetalKernels/elementwise``, in place on a ``GPUFrame``.
enum MetalElementwise {
    static func scale(_ c: MetalContext, _ x: GPUFrame, by s: Double) throws {
        try unary(c, "scale", x, s)
    }

    static func scaleLog10Guard(_ c: MetalContext, _ x: GPUFrame, scale s: Double = 1) throws {
        try unary(c, "scale_log10_guard", x, s)
    }

    static func exp10Scale(_ c: MetalContext, _ x: GPUFrame, scale s: Double = 1) throws {
        try unary(c, "exp10_scale", x, s)
    }

    static func affine3(
        _ c: MetalContext, _ x: GPUFrame, factor: [Double], offset: [Double]
    ) throws {
        precondition(x.channels == 3 && factor.count == 3 && offset.count == 3)
        var f = factor.map(Float.init)
        var o = offset.map(Float.init)
        var n = UInt32(x.count)
        try c.dispatch("affine3", count: x.count) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBytes(&f, length: 12, index: 1)
            e.setBytes(&o, length: 12, index: 2)
            e.setBytes(&n, length: 4, index: 3)
        }
    }

    static func matrix3(_ c: MetalContext, _ x: GPUFrame, _ m: Matrix3) throws {
        precondition(x.channels == 3)
        var values = [m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22].map(Float.init)
        var pixels = UInt32(x.pixelCount)
        try c.dispatch("matrix3", count: x.pixelCount) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBytes(&values, length: 36, index: 1)
            e.setBytes(&pixels, length: 4, index: 2)
        }
    }

    /// `a = a + amount * (a - b)`.
    static func unsharp(_ c: MetalContext, _ a: GPUFrame, _ b: GPUFrame, amount: Double) throws {
        try binary(c, "unsharp", a, b, amount)
    }

    /// `a = (1 - w) * a + w * b`.
    static func mix(_ c: MetalContext, _ a: GPUFrame, _ b: GPUFrame, weight: Double) throws {
        try binary(c, "mix_weighted", a, b, weight)
    }

    /// `a = b - a`.
    static func reverseSubtract(_ c: MetalContext, _ a: GPUFrame, _ b: GPUFrame) throws {
        precondition(a.count == b.count)
        var n = UInt32(a.count)
        try c.dispatch("reverse_subtract", count: a.count) { e in
            e.setBuffer(a.buffer, offset: 0, index: 0)
            e.setBuffer(b.buffer, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
        }
    }

    static func transfer(
        _ c: MetalContext, _ x: GPUFrame, _ function: TransferFunction, encode: Bool
    ) throws {
        guard function != .linear else { return }
        var code = UInt32(TransferFunction.allCases.firstIndex(of: function)!)
        var constants = [
            TransferFunction.bt2020Alpha, TransferFunction.bt2020Beta,
            TransferFunction.bt2020DecodeThreshold, TransferFunction.sRGBDecodeThreshold,
            TransferFunction.rommEt,
        ].map(Float.init)
        var direction = UInt32(encode ? 1 : 0)
        var n = UInt32(x.count)
        try c.dispatch("transfer", count: x.count) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBytes(&code, length: 4, index: 1)
            e.setBytes(&constants, length: constants.count * 4, index: 2)
            e.setBytes(&direction, length: 4, index: 3)
            e.setBytes(&n, length: 4, index: 4)
        }
    }

    private static func unary(
        _ c: MetalContext, _ kernel: String, _ x: GPUFrame, _ s: Double
    )
        throws
    {
        var scale = Float(s)
        var n = UInt32(x.count)
        try c.dispatch(kernel, count: x.count) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBytes(&scale, length: 4, index: 1)
            e.setBytes(&n, length: 4, index: 2)
        }
    }

    private static func binary(
        _ c: MetalContext, _ kernel: String, _ a: GPUFrame, _ b: GPUFrame, _ s: Double
    ) throws {
        precondition(a.count == b.count)
        var scale = Float(s)
        var n = UInt32(a.count)
        try c.dispatch(kernel, count: a.count) { e in
            e.setBuffer(a.buffer, offset: 0, index: 0)
            e.setBuffer(b.buffer, offset: 0, index: 1)
            e.setBytes(&scale, length: 4, index: 2)
            e.setBytes(&n, length: 4, index: 3)
        }
    }
}
#endif
