#if canImport(Metal)
import Foundation
import Metal

/// The filming-stage operators between the blurs, on ``GPUFrame``s.
enum MetalStage {

    /// `np.max` over the frame, NaN winning.
    static func maximum(_ c: MetalContext, _ x: GPUFrame) throws -> Float {
        let per = 4096
        let groups = max(1, (x.count + per - 1) / per)
        let partial = try GPUFrame(c, height: 1, width: groups, channels: 1)
        var n = UInt32(x.count)
        var chunk = UInt32(per)
        try c.dispatch("partial_max", count: groups) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBuffer(partial.buffer, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
            e.setBytes(&chunk, length: 4, index: 3)
        }
        var m = -Float.infinity
        let p = partial.floats
        for i in 0..<groups {
            if p[i].isNaN { return .nan }
            m = Swift.max(m, p[i])
        }
        return m
    }

    /// ``Diffusion/boostHighlights(_:boostEV:boostRange:protectEV:midgray:)``, in place.
    ///
    /// The curve constants are computed in float64 from the frame maximum, as on the CPU. A NaN
    /// maximum goes to the CPU operator, so the two backends fail the same way.
    static func boostHighlights(
        _ c: MetalContext, _ x: inout GPUFrame, boostEV: Double, boostRange: Double,
        protectEV: Double, midgray: Double = 0.184
    ) throws {
        if boostEV == 0 { return }
        let maxFloat = try maximum(c, x)
        if maxFloat.isNaN {
            var image = x.download()
            Diffusion.boostHighlights(
                &image, boostEV: boostEV, boostRange: boostRange, protectEV: protectEV,
                midgray: midgray)
            x = try GPUFrame(c, uploading: image)
            return
        }
        let maxRaw = Double(maxFloat)
        if maxRaw == 0 {
            memset(x.buffer.contents(), 0, x.count * MemoryLayout<Float>.stride)
            return
        }
        let rawX0 = Swift.min(Swift.max(midgray * pow(2.0, protectEV), 0.0), maxRaw)
        if rawX0 == maxRaw { return }
        let a = pow(28.0, 1.0 - boostRange)
        let x0 = rawX0 / maxRaw
        let denominator = exp(a * (1.0 - x0)) - a * (1.0 - x0) - 1.0
        let k = (pow(2.0, boostEV) - 1.0) / denominator
        var constants = SIMD4<Float>(Float(rawX0), Float(1.0 / maxRaw), Float(k * maxRaw), Float(a))
        var n = UInt32(x.count)
        try c.dispatch("boost_highlights", count: x.count) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBytes(&constants, length: 16, index: 1)
            e.setBytes(&n, length: 4, index: 2)
        }
    }

    /// ``Diffusion/applyHalation(_:_:pixelSizeMicrons:)``, in place, one channel at a time.
    static func halation(
        _ c: MetalContext, _ x: GPUFrame, _ halation: HalationParams, pixelSizeMicrons: Double
    ) throws {
        guard halation.active else { return }
        precondition(x.channels == 3)
        let triple = { (t: (Double, Double, Double)) in [t.0, t.1, t.2] }
        let pixels = x.pixelCount
        let plane = try GPUFrame(c, height: x.height, width: x.width, channels: 1)

        let amount = halation.scatterAmount
        let tailWeight = triple(halation.scatterTailWeight)
        let coreSigma = triple(halation.scatterCoreMicrons).map {
            $0 * halation.scatterSpatialScale / pixelSizeMicrons
        }
        let tailLambda = triple(halation.scatterTailMicrons).map {
            $0 * halation.scatterSpatialScale / pixelSizeMicrons
        }
        if amount > 0 && (coreSigma.contains { $0 > 0 } || tailLambda.contains { $0 > 0 }) {
            for channel in 0..<3 {
                try copyChannel(c, x, channel: channel, into: plane)
                let tail = try MetalBlur.exponential(
                    c, plane, decayPerChannel: [Swift.max(tailLambda[channel], 1e-6)])
                let core = try MetalBlur.gaussian(
                    c, plane, sigmaPerChannel: [Swift.max(coreSigma[channel], 1e-6)])
                var ch = UInt32(channel)
                var k = SIMD2<Float>(Float(amount), Float(tailWeight[channel]))
                var n = UInt32(pixels)
                try c.dispatch("halation_scatter", count: pixels) { e in
                    e.setBuffer(x.buffer, offset: 0, index: 0)
                    e.setBuffer(core.buffer, offset: 0, index: 1)
                    e.setBuffer(tail.buffer, offset: 0, index: 2)
                    e.setBytes(&ch, length: 4, index: 3)
                    e.setBytes(&k, length: 8, index: 4)
                    e.setBytes(&n, length: 4, index: 5)
                }
            }
        }

        let strength = triple(halation.halationStrength).map { $0 * halation.halationAmount }
        let firstSigma = triple(halation.halationFirstSigmaMicrons).map {
            $0 * halation.halationSpatialScale / pixelSizeMicrons
        }
        let bounces = halation.halationBounceCount
        guard bounces >= 1, strength.contains(where: { $0 > 0 }),
            firstSigma.contains(where: { $0 > 0 })
        else { return }
        var decay = (1...bounces).map { pow(halation.halationBounceDecay, Double($0 - 1)) }
        let decayTotal = decay.reduce(0, +)
        for i in decay.indices { decay[i] /= decayTotal }

        let accumulated = try GPUFrame(c, height: x.height, width: x.width, channels: 1)
        for channel in 0..<3 {
            try copyChannel(c, x, channel: channel, into: plane)
            memset(accumulated.buffer.contents(), 0, pixels * MemoryLayout<Float>.stride)
            for k in 1...bounces {
                let component = try MetalBlur.gaussian(
                    c, plane,
                    sigmaPerChannel: [Swift.max(firstSigma[channel] * Double(k).squareRoot(), 1e-6)])
                try MetalBlur.axpy(c, accumulated, component, weight: decay[k - 1])
            }
            var ch = UInt32(channel)
            var s = Float(strength[channel])
            var renormalise = UInt32(halation.halationRenormalize ? 1 : 0)
            var n = UInt32(pixels)
            try c.dispatch("halation_bounce", count: pixels) { e in
                e.setBuffer(x.buffer, offset: 0, index: 0)
                e.setBuffer(accumulated.buffer, offset: 0, index: 1)
                e.setBytes(&ch, length: 4, index: 2)
                e.setBytes(&s, length: 4, index: 3)
                e.setBytes(&renormalise, length: 4, index: 4)
                e.setBytes(&n, length: 4, index: 5)
            }
        }
    }

    /// ``Couplers/applyDensityCorrection``: the corrected log exposure, looked up again on the
    /// curves before couplers. Returns the new density; `density` is written over.
    static func couplerCorrection(
        _ c: MetalContext, density: GPUFrame, logRaw: GPUFrame, setup: Couplers.CorrectionSetup,
        tailWeight: Double, positive: Bool, before: MetalFilm.CurveTable
    ) throws -> GPUFrame {
        let m = setup.matrix
        var matrix = [m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22]
            .map(Float.init)
        var densityMax = SIMD3<Float>(
            Float(setup.densityMax[0]), Float(setup.densityMax[1]), Float(setup.densityMax[2]))
        var shift: Float = 0
        var isPositive = UInt32(positive ? 1 : 0)
        var pixels = UInt32(density.pixelCount)
        try c.dispatch("coupler_inhibitor", count: density.pixelCount) { e in
            e.setBuffer(density.buffer, offset: 0, index: 0)
            e.setBytes(&matrix, length: 36, index: 1)
            e.setBytes(&densityMax, length: 16, index: 2)
            e.setBytes(&shift, length: 4, index: 3)
            e.setBytes(&isPositive, length: 4, index: 4)
            e.setBytes(&pixels, length: 4, index: 5)
        }

        var inhibitor = density
        if setup.diffusionSizePixels > 0 {
            // Couplers.diffuseInPlace: the Gaussian core and the exponential tail, blended.
            let sigma = setup.diffusionSizePixels
            let decay = setup.diffusionTailPixels
            let tail = try MetalBlur.exponential(
                c, inhibitor, decayPerChannel: [Double](repeating: decay, count: 3))
            let core = try MetalBlur.gaussian(
                c, inhibitor, sigmaPerChannel: [Double](repeating: sigma, count: 3))
            try MetalElementwise.mix(c, core, tail, weight: tailWeight)
            inhibitor = core
        }
        // raw - inhibitor, then the lookup on the curves before couplers.
        try MetalElementwise.reverseSubtract(c, inhibitor, logRaw)
        try MetalFilm.interpolate(c, inhibitor, into: inhibitor, table: before)
        return inhibitor
    }

    static func copyChannel(
        _ c: MetalContext, _ x: GPUFrame, channel: Int, into plane: GPUFrame
    ) throws {
        var p = MetalBlur.Plane(
            height: UInt32(x.height), width: UInt32(x.width), srcStride: UInt32(x.channels),
            srcOffset: UInt32(channel), dstStride: 1, dstOffset: 0)
        try c.dispatch("copy_channel", count: x.pixelCount) { e in
            e.setBuffer(x.buffer, offset: 0, index: 0)
            e.setBuffer(plane.buffer, offset: 0, index: 1)
            e.setBytes(&p, length: MemoryLayout<MetalBlur.Plane>.stride, index: 2)
        }
    }
}
#endif
