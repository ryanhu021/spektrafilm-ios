import Foundation

/// The public entry point.
///
/// Mirrors `spektrafilm.runtime.process.Simulator`. Construct one per parameter set and reuse it
/// across frames. Construction computes the per-film spectral LUT and the midgray references. Both
/// are expensive, and neither depends on the image.
///
/// ```swift
/// let params = try RuntimePhotoParams.make(
///     film: "kodak_portra_400", print: "kodak_portra_endura")
/// let simulator = try Simulator(params)
/// let rendered = try simulator.process(image)
/// ```
public final class Simulator {
    private let pipeline: SimulationPipeline

    /// - Parameter backend: ``ComputeBackend/cpu`` renders the parity-tested float64 result.
    ///   ``ComputeBackend/metal`` moves the operators that have a GPU version to float32 on the GPU.
    public init(
        _ params: RuntimePhotoParams, resampler: any Resampler = UnavailableResampler(),
        backend: ComputeBackend = .cpu
    )
        throws
    {
        pipeline = try SimulationPipeline(params: params, resampler: resampler, backend: backend)
    }

    /// Runs the whole pipeline: scene-linear RGB in, output-encoded RGB out.
    public func process(_ image: ImageBuffer) throws -> ImageBuffer {
        try pipeline.process(image)
    }

    /// Runs part of the pipeline, entering or leaving at a named tap.
    ///
    /// Useful for inspecting an intermediate, and for the per-stage parity fixtures.
    public func process(_ image: ImageBuffer, inject: Tap?, collect: Tap?) throws -> ImageBuffer {
        try pipeline.process(image, inject: inject, collect: collect)
    }

    /// Renders float32 RGB, interleaved `[height][width][3]`, end to end.
    ///
    /// `fill` writes the input pixels and `read` receives the output with its dimensions. On the
    /// Metal backend the input goes straight into GPU-visible memory and the output is read from
    /// it, so no float64 copy of the frame is ever made: at 12 MP that is 576 MB less peak memory
    /// than ``process(_:)``. Otherwise the pixels go through an ``ImageBuffer``.
    public func processFloat<T>(
        height: Int, width: Int,
        fill: (UnsafeMutableBufferPointer<Float>) throws -> Void,
        read: (UnsafeBufferPointer<Float>, _ height: Int, _ width: Int) throws -> T
    ) throws -> T {
        #if canImport(Metal)
        if let metal = pipeline.floatPipeline, let context = MetalContext.shared {
            var input: GPUFrame? = try GPUFrame(context, height: height, width: width, channels: 3)
            try fill(UnsafeMutableBufferPointer(start: input!.floats, count: height * width * 3))
            let out = try pipeline.process(taking: &input, collect: .rgbOut, pipeline: metal)
            return try read(
                UnsafeBufferPointer(start: out.floats, count: out.count), out.height, out.width)
        }
        #endif
        var floats = [Float](repeating: 0, count: height * width * 3)
        try floats.withUnsafeMutableBufferPointer { try fill($0) }
        let image = ImageBuffer(
            height: height, width: width, channels: 3, values: floats.map(Double.init))
        floats = []
        let out = try pipeline.process(image)
        let result = out.values.map(Float.init)
        return try result.withUnsafeBufferPointer { try read($0, out.height, out.width) }
    }

    /// Whether ``processFloat(height:width:fill:read:)`` runs on the GPU, which sets its memory
    /// cost: ``RenderBudget/metalBytesPerMegapixel`` rather than
    /// ``RenderBudget/bytesPerMegapixel``.
    public var rendersFloatOnGPU: Bool {
        #if canImport(Metal)
        return pipeline.floatPipeline != nil && MetalContext.shared != nil
        #else
        return false
        #endif
    }

    /// Per-node wall-clock times from the last run.
    public var timings: [String: TimeInterval] { pipeline.timings }

    /// Total wall-clock time of the last run.
    public var elapsed: TimeInterval? { pipeline.elapsed }

    public func formattedTimings() -> String { pipeline.formattedTimings() }
}

extension Simulator {
    /// One-shot convenience, matching the reference's `simulate(image, params)`.
    ///
    /// Builds a simulator, runs one frame, and throws it away. For more than one frame, keep a
    /// ``Simulator``: construction computes the spectral LUT.
    public static func simulate(
        _ image: ImageBuffer,
        params: RuntimePhotoParams,
        resampler: any Resampler = UnavailableResampler()
    ) throws -> ImageBuffer {
        try Simulator(params, resampler: resampler).process(image)
    }
}
