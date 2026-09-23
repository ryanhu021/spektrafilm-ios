#if canImport(Metal)
import Foundation
import Metal

/// The spectral upsampling and the density-curve lookup on ``GPUFrame``s.
enum MetalFilm {

    /// ``Hanatos2025RawConverter/raw(rgb:)``. `lut` is the converter's `tc_lut`, uploaded once.
    static func rgbToRaw(
        _ c: MetalContext, _ rgb: GPUFrame, converter: Hanatos2025RawConverter, lut: MTLBuffer,
        lutSize: Int
    ) throws -> GPUFrame {
        precondition(rgb.channels == 3)
        let out = try GPUFrame(c, height: rgb.height, width: rgb.width, channels: 3)
        let mm = converter.matrix
        var m = [mm.m00, mm.m01, mm.m02, mm.m10, mm.m11, mm.m12, mm.m20, mm.m21, mm.m22]
            .map(Float.init)
        var code = UInt32(TransferFunction.allCases.firstIndex(of: converter.transfer)!)
        var k = MetalElementwise.transferConstants
        var decode = UInt32(converter.applyCCTFDecoding ? 1 : 0)
        var size = UInt32(lutSize)
        var pixels = UInt32(rgb.pixelCount)
        try c.dispatch("rgb_to_raw", count: rgb.pixelCount) { e in
            e.setBuffer(rgb.buffer, offset: 0, index: 0)
            e.setBuffer(lut, offset: 0, index: 1)
            e.setBuffer(out.buffer, offset: 0, index: 2)
            e.setBytes(&m, length: 36, index: 3)
            e.setBytes(&code, length: 4, index: 4)
            e.setBytes(&k, length: k.count * 4, index: 5)
            e.setBytes(&decode, length: 4, index: 6)
            e.setBytes(&size, length: 4, index: 7)
            e.setBytes(&pixels, length: 4, index: 8)
        }
        return out
    }

    /// A density-curve table ready for ``interpolate(_:_:into:table:)``.
    struct CurveTable {
        let axis: MTLBuffer
        let values: MTLBuffer
        let inverseWidths: MTLBuffer
        let count: Int
        let perChannel: Bool

        /// `axis` is shared, or per channel and interleaved, as ``Interpolation/fastInterp``
        /// takes it. The reciprocal widths are computed in float64 and rounded once, as the CPU
        /// computes them.
        init(_ c: MetalContext, axis: [Double], values: [Double]) throws {
            perChannel = axis.count == values.count
            count = perChannel ? values.count / 3 : axis.count
            let stride = perChannel ? 3 : 1
            var inverse = [Double](repeating: 0, count: (count - 1) * stride)
            for s in 0..<stride {
                for i in 0..<(count - 1) {
                    let d = axis[(i + 1) * stride + s] - axis[i * stride + s]
                    inverse[i * stride + s] = d != 0 ? 1.0 / d : 0.0
                }
            }
            self.axis = try c.buffer(from: axis)
            self.values = try c.buffer(from: values)
            inverseWidths = try c.buffer(from: inverse)
        }

        /// ``DensityCurves/densityFromLogExposure(logExposure:curves:axis:gammaFactor:)``'s
        /// table: the shared axis divided by each channel's gamma.
        init(
            _ c: MetalContext, curves: [Double], logExposure: [Double],
            gamma: (Double, Double, Double)
        ) throws {
            let g = [gamma.0, gamma.1, gamma.2]
            var axis = [Double](repeating: 0, count: logExposure.count * 3)
            for i in logExposure.indices {
                for ch in 0..<3 { axis[i * 3 + ch] = logExposure[i] / g[ch] }
            }
            try self.init(c, axis: axis, values: curves)
        }
    }

    /// ``Interpolation/fastInterp(_:axis:values:)``. `out` may be `x` itself.
    static func interpolate(
        _ c: MetalContext, _ x: GPUFrame, into out: GPUFrame, table: CurveTable
    ) throws {
        precondition(x.channels == 3 && out.count == x.count)
        var count = UInt32(table.count)
        var perChannel = UInt32(table.perChannel ? 1 : 0)
        var n = UInt32(x.count)
        try c.dispatch("fast_interp", count: x.count) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBuffer(out.buffer, offset: 0, index: 1)
            e.setBuffer(table.axis, offset: 0, index: 2)
            e.setBuffer(table.values, offset: 0, index: 3)
            e.setBuffer(table.inverseWidths, offset: 0, index: 4)
            e.setBytes(&count, length: 4, index: 5)
            e.setBytes(&perChannel, length: 4, index: 6)
            e.setBytes(&n, length: 4, index: 7)
        }
    }
}
#endif
