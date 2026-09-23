#if canImport(Metal)
import Foundation
import Metal

/// ``Grain`` on ``GPUFrame``s, with ``MetalBlur`` standing in for the ``SpatialFilter``.
///
/// Each pixel draws from the same Philox stream as on the CPU: the same ``PhiloxKey`` fields and
/// the linear pixel index as the counter, so the result does not depend on thread scheduling. The
/// GPU's uniforms carry 24 bits where the CPU's carry 53, so the draws agree in distribution, and in
/// value at nearly every pixel.
enum MetalGrain {

    /// Mirrors `GrainLayer` in ``MetalKernels/grain``. Every `SIMD2<Float>` is a double-float
    /// `(hi, lo)` pair.
    struct Layer {
        var seed: UInt64
        var densityMin: SIMD2<Float>
        var densityMax: SIMD2<Float>
        var particles: SIMD2<Float>
        var uniformity: SIMD2<Float>
        var odParticle: SIMD2<Float>
        var saturationScale = MetalGrain.split(Grain.saturationScale)
        var probabilityFloor = MetalGrain.split(Grain.probabilityFloor)
        var probabilityCeiling = MetalGrain.split(Grain.probabilityCeiling)
        var channel: UInt32
        var stream: UInt32
        var pixels: UInt32
        var steps: UInt32 = 0
        var positive: UInt32 = 0
        var interpolate: UInt32 = 0
        var dstStride: UInt32 = 1
        var dstOffset: UInt32 = 0
        var accumulate: UInt32 = 0

        init(
            seed: UInt64, channel: Int, stream: Int, pixels: Int, densityMin: Double,
            densityMax: Double, particles: Double, uniformity: Double
        ) {
            self.seed = seed
            self.channel = UInt32(channel)
            self.stream = UInt32(stream)
            self.pixels = UInt32(pixels)
            self.densityMin = MetalGrain.split(densityMin)
            self.densityMax = MetalGrain.split(densityMax)
            self.particles = MetalGrain.split(particles)
            self.uniformity = MetalGrain.split(uniformity)
            // As Grain.layerParticleModel computes it, in float64.
            odParticle = MetalGrain.split(densityMax / particles)
        }
    }

    /// The per-channel axes, the nine sublayer curves and the reciprocal interval widths that
    /// ``Grain/sublayerPlane(_:channel:sublayer:densityCurves:densityCurvesLayers:positive:)``
    /// interpolates with, each row `steps` long. Positive stocks have the axis negated.
    struct SublayerTables {
        let axis: MTLBuffer
        let curves: MTLBuffer
        let inverseWidths: MTLBuffer
        let steps: Int

        init(
            _ c: MetalContext, densityCurves: [Double], densityCurvesLayers: [Double],
            positive: Bool
        ) throws {
            let steps = densityCurves.count / 3
            precondition(steps >= 2, "axis needs at least two samples")
            precondition(densityCurvesLayers.count == steps * 9)
            var axis = [Double](repeating: 0, count: 3 * steps)
            var inverse = [Double](repeating: 0, count: 3 * steps)
            var curves = [Double](repeating: 0, count: 9 * steps)
            for channel in 0..<3 {
                for s in 0..<steps {
                    let a = densityCurves[s * 3 + channel]
                    axis[channel * steps + s] = positive ? -a : a
                }
                for s in 0..<(steps - 1) {
                    let d = axis[channel * steps + s + 1] - axis[channel * steps + s]
                    inverse[channel * steps + s] = d != 0 ? 1.0 / d : 0.0
                }
            }
            for i in 0..<9 {
                for s in 0..<steps { curves[i * steps + s] = densityCurvesLayers[s * 9 + i] }
            }
            self.steps = steps
            self.axis = try c.buffer(from: axis)
            self.curves = try c.buffer(from: curves)
            inverseWidths = try c.buffer(from: inverse)
        }
    }

    /// `Grain.apply`, with the same inputs less the spatial filter.
    ///
    /// Returns `density` itself when grain is off or bypassed.
    static func apply(
        _ c: MetalContext,
        _ density: GPUFrame,
        pixelSizeMicrons: Double,
        params: GrainParams,
        densityCurves: [Double],
        densityCurvesLayers: [Double],
        positive: Bool,
        seed: UInt64 = 0,
        bypass: Bool = false
    ) throws -> GPUFrame {
        guard params.active, !bypass else { return density }
        precondition(density.channels == 3, "density must have 3 channels")

        if !params.sublayersActive {
            let derived = Grain.SingleLayerParameters(
                params: params, pixelSizeMicrons: pixelSizeMicrons,
                densityMaxCurves: nanMax(densityCurves, channels: 3))
            return try applyToDensity(c, density, derived: derived, seed: seed)
        }

        let derived = Grain.LayeredParameters(
            params: params, pixelSizeMicrons: pixelSizeMicrons,
            densityMaxLayers: nanMax(densityCurvesLayers, channels: 9))
        let tables = try SublayerTables(
            c, densityCurves: densityCurves, densityCurvesLayers: densityCurvesLayers,
            positive: positive)
        let grain = try accumulateLayers(
            c, density, derived: derived, tables: tables, positive: positive, seed: seed)
        return try finishLayers(c, grain, derived: derived, seed: seed)
    }

    /// The layer description for one `(channel, sublayer)` plane of the layered path.
    static func layer(
        _ derived: Grain.LayeredParameters, tables: SublayerTables, positive: Bool,
        channel: Int, sublayer: Int, pixels: Int, seed: UInt64
    ) -> Layer {
        let i = sublayer * 3 + channel
        var layer = Layer(
            seed: seed, channel: channel, stream: sublayer, pixels: pixels,
            densityMin: derived.densityMinLayers[i], densityMax: derived.densityMaxLayers[i],
            particles: derived.particlesPerPixel[i], uniformity: derived.uniformity[channel])
        layer.steps = UInt32(tables.steps)
        layer.positive = positive ? 1 : 0
        layer.interpolate = 1
        return layer
    }

    static func accumulateLayers(
        _ c: MetalContext, _ density: GPUFrame, derived: Grain.LayeredParameters,
        tables: SublayerTables, positive: Bool, seed: UInt64
    ) throws -> GPUFrame {
        let out = try zeroFrame(c, like: density)
        let plane =
            derived.blurDyeClouds > 0
            ? try GPUFrame(c, height: density.height, width: density.width, channels: 1) : nil
        for channel in 0..<3 {
            for sublayer in 0..<Grain.sublayerCount {
                var layer = layer(
                    derived, tables: tables, positive: positive, channel: channel,
                    sublayer: sublayer, pixels: density.pixelCount, seed: seed)
                guard let plane else {
                    layer.dstStride = 3
                    layer.dstOffset = UInt32(channel)
                    layer.accumulate = 1
                    try draw(c, density, into: out, layer: &layer, tables: tables)
                    continue
                }
                try draw(c, density, into: plane, layer: &layer, tables: tables)
                // The gate is on the parameter, not on the sigma it produces, as on the CPU.
                let i = sublayer * 3 + channel
                let sigma =
                    derived.blurDyeClouds
                    * (derived.densityMaxLayers[i] / derived.particlesPerPixel[i]).squareRoot()
                let blurred = try MetalBlur.gaussian(c, plane, sigmaPerChannel: [sigma])
                try accumulate(c, blurred, into: out, channel: channel)
            }
        }
        return out
    }

    /// The clumping field, the fog subtraction and the closing blur, in that order.
    static func finishLayers(
        _ c: MetalContext, _ grain: GPUFrame, derived: Grain.LayeredParameters, seed: UInt64
    ) throws -> GPUFrame {
        let out = try addMicroStructure(
            c, grain, microStructure: derived.microStructure,
            pixelSizeMicrons: derived.pixelSizeMicrons, seed: seed)
        try MetalElementwise.affine3(
            c, out, factor: [1, 1, 1], offset: derived.densityMin.map { -$0 })
        guard derived.blurSigmaPixels > 0 else { return out }
        return try MetalBlur.gaussian(
            c, out, sigmaPerChannel: [Double](repeating: derived.blurSigmaPixels, count: 3))
    }

    /// ``Grain/applyToDensity(_:derived:seed:spatial:)``.
    static func applyToDensity(
        _ c: MetalContext, _ density: GPUFrame, derived: Grain.SingleLayerParameters,
        seed: UInt64
    ) throws -> GPUFrame {
        let out = try zeroFrame(c, like: density)
        for channel in 0..<3 {
            for repeatIndex in 0..<derived.subLayerCount {
                var layer = Layer(
                    seed: seed, channel: channel, stream: repeatIndex, pixels: density.pixelCount,
                    densityMin: derived.densityMin[channel],
                    densityMax: derived.densityMax[channel],
                    particles: derived.particlesPerPixel[channel],
                    uniformity: derived.uniformity[channel])
                layer.dstStride = 3
                layer.dstOffset = UInt32(channel)
                layer.accumulate = 1
                try draw(c, density, into: out, layer: &layer, tables: nil)
            }
        }
        let inverse = 1.0 / Double(derived.subLayerCount)
        try MetalElementwise.affine3(
            c, out, factor: [inverse, inverse, inverse], offset: derived.densityMin.map { -$0 })
        // The threshold here is `> 0.4`, and the layered path's is `> 0`, as on the CPU.
        guard derived.blurSigmaPixels > 0.4 else { return out }
        return try MetalBlur.gaussian(
            c, out, sigmaPerChannel: [Double](repeating: derived.blurSigmaPixels, count: 3))
    }

    /// ``Grain/addMicroStructure(_:microStructure:pixelSizeMicrons:seed:spatial:)``, in place when
    /// the gate is shut.
    static func addMicroStructure(
        _ c: MetalContext, _ image: GPUFrame, microStructure: (Double, Double),
        pixelSizeMicrons: Double, seed: UInt64
    ) throws -> GPUFrame {
        let blurPixels = microStructure.0 / pixelSizeMicrons
        let sigma = microStructure.1 * 0.001 / pixelSizeMicrons
        guard sigma > 0.05 else { return image }
        precondition(image.channels == 3)

        var clumping = try GPUFrame(c, height: image.height, width: image.width, channels: 3)
        let log = Distributions.lognormalLogParameters(mean: 1.0, std: sigma)
        var logParameters = SIMD2<Float>(
            Float(log.mu), log.sigma < Distributions.lognormalSigmaFloor ? 0 : Float(log.sigma))
        var s = seed
        var stream = UInt32(Grain.microStructureStream)
        var pixels = UInt32(image.pixelCount)
        try c.dispatch("grain_clumping", count: image.count) { e in
            e.setBuffer(clumping.buffer, offset: 0, index: 0)
            e.setBytes(&s, length: 8, index: 1)
            e.setBytes(&logParameters, length: 8, index: 2)
            e.setBytes(&stream, length: 4, index: 3)
            e.setBytes(&pixels, length: 4, index: 4)
        }
        if blurPixels > 0.4 {
            clumping = try MetalBlur.gaussian(
                c, clumping, sigmaPerChannel: [blurPixels, blurPixels, blurPixels])
        }
        var n = UInt32(image.count)
        try c.dispatch("grain_multiply", count: image.count) { e in
            e.setBuffer(image.buffer, offset: 0, index: 0)
            e.setBuffer(clumping.buffer, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
        }
        return image
    }

    // MARK: - Dispatch

    /// One particle population. `tables` is `nil` on the single-layer path, which reads the
    /// channel density directly.
    static func draw(
        _ c: MetalContext, _ density: GPUFrame, into dst: GPUFrame, layer: inout Layer,
        tables: SublayerTables?
    ) throws {
        try encodeLayer(c, "grain_layer", density, dst.buffer, layer: &layer, tables: tables)
    }

    /// Runs `kernel` with the argument layout `grain_layer` and `grain_layer_setup` share.
    static func encodeLayer(
        _ c: MetalContext, _ kernel: String, _ density: GPUFrame, _ dst: MTLBuffer,
        layer: inout Layer, tables: SublayerTables?
    ) throws {
        let count = density.pixelCount
        try c.dispatch(kernel, count: count) { e in
            e.setBuffer(density.buffer, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBuffer(tables?.axis ?? density.buffer, offset: 0, index: 2)
            e.setBuffer(tables?.curves ?? density.buffer, offset: 0, index: 3)
            e.setBuffer(tables?.inverseWidths ?? density.buffer, offset: 0, index: 4)
            e.setBytes(&layer, length: MemoryLayout<Layer>.stride, index: 5)
        }
    }

    static func accumulate(
        _ c: MetalContext, _ plane: GPUFrame, into out: GPUFrame, channel: Int
    ) throws {
        var ch = UInt32(channel)
        var pixels = UInt32(plane.pixelCount)
        try c.dispatch("grain_accumulate", count: plane.pixelCount) { e in
            e.setBuffer(out.buffer, offset: 0, index: 0)
            e.setBuffer(plane.buffer, offset: 0, index: 1)
            e.setBytes(&ch, length: 4, index: 2)
            e.setBytes(&pixels, length: 4, index: 3)
        }
    }

    /// A float64 value as a double-float `(hi, lo)` pair, about 48 significant bits.
    static func split(_ value: Double) -> SIMD2<Float> {
        let hi = Float(value)
        return SIMD2(hi, Float(value - Double(hi)))
    }

    static func zeroFrame(_ c: MetalContext, like frame: GPUFrame) throws -> GPUFrame {
        let out = try GPUFrame(c, height: frame.height, width: frame.width, channels: 3)
        memset(out.buffer.contents(), 0, out.count * MemoryLayout<Float>.stride)
        return out
    }
}
#endif
