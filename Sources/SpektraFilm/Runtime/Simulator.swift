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
