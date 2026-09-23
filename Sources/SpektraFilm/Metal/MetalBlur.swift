#if canImport(Metal)
import Foundation
import Metal

/// ``GaussianFilter`` and ``ExponentialFilter`` on ``GPUFrame``s.
enum MetalBlur {
    /// Mirrors the `Plane` struct in ``MetalKernels/blur``.
    struct Plane {
        var height: UInt32
        var width: UInt32
        var srcStride: UInt32
        var srcOffset: UInt32
        var dstStride: UInt32
        var dstOffset: UInt32
    }

    /// ``GaussianFilter/apply(_:sigmaPerChannel:truncate:)``, into a new frame.
    ///
    /// Each channel goes through one single-channel scratch plane, and takes the FIR path below
    /// sigma 3 and the IIR path at and above it, as ``GaussianFilter/filterPlane`` dispatches.
    static func gaussian(
        _ c: MetalContext, _ input: GPUFrame, sigmaPerChannel: [Double],
        truncate: Double = GaussianFilter.defaultTruncate
    ) throws -> GPUFrame {
        precondition(sigmaPerChannel.count == input.channels)
        let out = try GPUFrame(c, height: input.height, width: input.width, channels: input.channels)
        let scratch = try GPUFrame(c, height: input.height, width: input.width, channels: 1)
        for (channel, sigma) in sigmaPerChannel.enumerated() {
            try blurChannel(
                c, input, into: out, channel: channel, sigma: sigma, truncate: truncate,
                scratch: scratch)
        }
        return out
    }

    /// ``ExponentialFilter/apply(_:decayPerChannel:mixture:truncate:)``: a weighted sum of
    /// Gaussians, accumulated in the fit's order.
    static func exponential(
        _ c: MetalContext, _ input: GPUFrame, decayPerChannel: [Double],
        mixture: ExponentialFilter.MixtureSize = .three,
        truncate: Double = GaussianFilter.defaultTruncate
    ) throws -> GPUFrame {
        let result = try GPUFrame(
            c, height: input.height, width: input.width, channels: input.channels)
        memset(result.buffer.contents(), 0, result.count * MemoryLayout<Float>.stride)
        for (amplitude, ratio) in ExponentialFilter.fit(mixture) {
            let component = try gaussian(
                c, input, sigmaPerChannel: decayPerChannel.map { ratio * $0 }, truncate: truncate)
            try axpy(c, result, component, weight: amplitude)
        }
        return result
    }

    /// `a += weight * b`.
    static func axpy(_ c: MetalContext, _ a: GPUFrame, _ b: GPUFrame, weight: Double) throws {
        precondition(a.count == b.count)
        var w = Float(weight)
        var n = UInt32(a.count)
        try c.dispatch("axpy", count: a.count) { e in
            e.setBuffer(a.buffer, offset: 0, index: 0)
            e.setBuffer(b.buffer, offset: 0, index: 1)
            e.setBytes(&w, length: 4, index: 2)
            e.setBytes(&n, length: 4, index: 3)
        }
    }

    static func blurChannel(
        _ c: MetalContext, _ input: GPUFrame, into out: GPUFrame, channel: Int, sigma: Double,
        truncate: Double, scratch: GPUFrame
    ) throws {
        let h = UInt32(input.height)
        let w = UInt32(input.width)
        let stride = UInt32(input.channels)
        let ch = UInt32(channel)
        var toScratch = Plane(
            height: h, width: w, srcStride: stride, srcOffset: ch, dstStride: 1, dstOffset: 0)
        var toOut = Plane(
            height: h, width: w, srcStride: 1, srcOffset: 0, dstStride: stride, dstOffset: ch)
        let planeSize = MemoryLayout<Plane>.stride

        if sigma <= 0 {
            var direct = Plane(
                height: h, width: w, srcStride: stride, srcOffset: ch, dstStride: stride,
                dstOffset: ch)
            try c.dispatch("copy_channel", count: input.pixelCount) { e in
                e.setBuffer(input.buffer, offset: 0, index: 0)
                e.setBuffer(out.buffer, offset: 0, index: 1)
                e.setBytes(&direct, length: planeSize, index: 2)
            }
            return
        }

        if sigma >= GaussianFilter.smallSigmaMax {
            let k = GaussianFilter.yvvCoefficients(sigma: sigma)
            // Each coefficient as a double-float (hi, lo) pair, for the kernels' recursion.
            var coefficients = [k.b, k.b1, k.b2, k.b3].flatMap { v -> [Float] in
                let hi = Float(v)
                return [hi, Float(v - Double(hi))]
            }
            try c.dispatch("iir_rows", count: input.height) { e in
                e.setBuffer(input.buffer, offset: 0, index: 0)
                e.setBuffer(scratch.buffer, offset: 0, index: 1)
                e.setBytes(&coefficients, length: 32, index: 2)
                e.setBytes(&toScratch, length: planeSize, index: 3)
            }
            try c.dispatch("iir_columns", count: input.width) { e in
                e.setBuffer(scratch.buffer, offset: 0, index: 0)
                e.setBuffer(out.buffer, offset: 0, index: 1)
                e.setBytes(&coefficients, length: 32, index: 2)
                e.setBytes(&toOut, length: planeSize, index: 3)
            }
            return
        }

        let (kernel, radius) = GaussianFilter.kernel1D(sigma: sigma, truncate: truncate)
        var weights = kernel.map(Float.init)
        var r = Int32(radius)
        try c.dispatch("fir_vertical", width: input.width, height: input.height) { e in
            e.setBuffer(input.buffer, offset: 0, index: 0)
            e.setBuffer(scratch.buffer, offset: 0, index: 1)
            e.setBytes(&weights, length: weights.count * 4, index: 2)
            e.setBytes(&r, length: 4, index: 3)
            e.setBytes(&toScratch, length: planeSize, index: 4)
        }
        try c.dispatch("fir_horizontal", width: input.width, height: input.height) { e in
            e.setBuffer(scratch.buffer, offset: 0, index: 0)
            e.setBuffer(out.buffer, offset: 0, index: 1)
            e.setBytes(&weights, length: weights.count * 4, index: 2)
            e.setBytes(&r, length: 4, index: 3)
            e.setBytes(&toOut, length: planeSize, index: 4)
        }
    }
}
#endif
