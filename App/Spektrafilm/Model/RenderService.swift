import CoreGraphics
import Foundation
import SpektraFilm

/// Runs the engine off the main thread, one render at a time.
///
/// An actor because `Simulator` is a class holding the per-film spectral LUT and the midgray
/// references, so it is not `Sendable` and must not be touched from two tasks at once. Serialising
/// here also means a queued render cannot start while an earlier one is still mutating buffers.
///
/// The simulator is cached and rebuilt only when the parameters change, since construction costs
/// about 20 ms: measurable against an 83 ms scrub render, wasteful to repeat.
actor RenderService {
    /// Which of the three measured budgets a request is asking for.
    enum Quality: Sendable, Comparable {
        /// 320 px, preview mode. Measured at 83 ms, so roughly 12 fps while a control is moving.
        case scrub
        /// 640 px, preview mode. Measured at 343 ms, for when the hand comes off the control.
        case settle
        /// 640 px with grain and the spatial effects. Measured at 623 ms.
        case proof
        /// Full resolution, everything on. Roughly 2.3 s per megapixel.
        case full

        var longEdge: Int? {
            switch self {
            case .scrub: return 320
            case .settle, .proof: return 640
            case .full: return nil
            }
        }

        /// Ordered by how much work they ask for, so a coalescing scheduler can keep the larger of
        /// two pending requests.
        private var rank: Int {
            switch self {
            case .scrub: return 0
            case .settle: return 1
            case .proof: return 2
            case .full: return 3
            }
        }

        static func < (a: Quality, b: Quality) -> Bool { a.rank < b.rank }

        /// Preview mode drops grain and the expensive blurs while keeping the halation kernel widths.
        var previewMode: Bool {
            switch self {
            case .scrub, .settle: return true
            case .proof, .full: return false
            }
        }
    }

    struct Result: Sendable {
        let image: CGImage
        let quality: Quality
        let generation: UInt64
        let milliseconds: Double
        /// Per-stage timings, longest first, for the diagnostics panel.
        let stages: [(String, Double)]
    }

    private var simulator: Simulator?
    private var simulatorParams: RuntimePhotoParams?

    /// Renders `source` and returns an image, or `nil` if a newer generation superseded this request
    /// before it started.
    ///
    /// `generation` is the caller's monotonic counter. Passing `currentGeneration` lets a request that
    /// has already been overtaken be dropped without doing the work.
    func render(
        source: CGImage,
        params: RuntimePhotoParams,
        quality: Quality,
        tap: Tap,
        generation: UInt64,
        currentGeneration: @Sendable () async -> UInt64
    ) async throws -> Result? {
        if await currentGeneration() != generation { return nil }

        var params = params
        params.settings.previewMode = quality.previewMode

        let buffer = try ImageBridge.buffer(source: source, quality: quality)
        if await currentGeneration() != generation { return nil }

        let simulator = try simulator(for: params)
        let started = DispatchTime.now().uptimeNanoseconds
        let rendered = try simulator.process(buffer, inject: nil, collect: tap)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6

        // Intermediate taps are densities or log exposures, not display values. Normalise them so the
        // negative is legible rather than a black rectangle, and say so in the UI.
        let display = tap == .rgbOut ? rendered : Self.normaliseForDisplay(rendered, tap: tap)
        let space =
            tap == .rgbOut
            ? ImageBridge.colourSpace(forOutput: params.io.outputColourSpace)
            : CGColorSpace(name: CGColorSpace.sRGB)!
        let image = try ImageBridge.image(from: display, colourSpace: space)

        return Result(
            image: image,
            quality: quality,
            generation: generation,
            milliseconds: elapsed,
            stages: simulator.timings.sorted { $0.value > $1.value }.map { ($0.key, $0.value * 1000) }
        )
    }

    private func simulator(for params: RuntimePhotoParams) throws -> Simulator {
        if let simulator, simulatorParams == params { return simulator }
        // Auto-exposure meters on a downsampled preview, so the resampler is not optional.
        let built = try Simulator(params, resampler: SkimageResampler())
        simulator = built
        simulatorParams = params
        return built
    }

    /// Maps a density or log-exposure tap into something viewable.
    ///
    /// Densities run roughly 0 to 3 and log exposures span negative values, so neither is meaningful
    /// as display RGB. This inverts density (so the negative reads as a negative, orange mask and all)
    /// and otherwise scales to the observed range.
    private static func normaliseForDisplay(_ buffer: ImageBuffer, tap: Tap) -> ImageBuffer {
        var out = buffer
        let isDensity = tap == .cmyFilm || tap == .cmyPrint
        var lo = Double.infinity
        var hi = -Double.infinity
        for v in buffer.values where v.isFinite {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        guard hi > lo else { return out }
        let span = hi - lo
        for i in out.values.indices {
            let v = out.values[i]
            let unit = v.isFinite ? (v - lo) / span : 0
            out.values[i] = isDensity ? 1 - unit : unit
        }
        return out
    }
}

extension ImageBridge {
    /// Decodes at the size a quality tier asks for.
    static func buffer(source: CGImage, quality: RenderService.Quality) throws -> ImageBuffer {
        try buffer(from: source, longEdge: quality.longEdge)
    }
}
