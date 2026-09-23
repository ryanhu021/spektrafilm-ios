import CoreGraphics
import Foundation
import SpektraFilm

/// Runs the engine off the main thread, one render at a time.
///
/// `Simulator` holds the per-film spectral LUT and the midgray references and is not `Sendable`, so
/// only this actor touches it. The simulator is cached and rebuilt when the parameters change, which
/// costs about 1.5 ms.
actor RenderService {
    /// Which of the three measured budgets a request is asking for.
    enum Quality: Sendable, Comparable {
        /// 320 px, preview mode, while a control is moving. 19 ms on an M4 Pro.
        case scrub
        /// 640 px, preview mode, for when the control is released. 57 ms on an M4 Pro.
        case settle
        /// 640 px with grain and the spatial effects. 94 ms on an M4 Pro.
        case proof
        /// Everything on, at the largest size ``RenderBudget`` allows.
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
        let tap: Tap
        let milliseconds: Double
        /// Per-stage timings, longest first, for the diagnostics panel.
        let stages: [(String, Double)]
        /// What was actually rendered, which for `.full` may be smaller than the source.
        let pixelSize: (width: Int, height: Int)
        /// True when the memory budget forced a smaller frame than the source.
        let wasDownscaled: Bool
    }

    private var simulator: Simulator?
    private var simulatorParams: RuntimePhotoParams?

    func render(
        source: CGImage,
        params: RuntimePhotoParams,
        quality: Quality,
        tap: Tap
    ) throws -> Result {
        var params = params
        params.settings.previewMode = quality.previewMode

        let cap =
            quality == .full
            ? RenderBudget.longEdge(forWidth: source.width, height: source.height)
            : quality.longEdge
        let buffer = try ImageBridge.buffer(from: source, longEdge: cap)

        let simulator = try simulator(for: params)
        let started = DispatchTime.now().uptimeNanoseconds
        let rendered = try simulator.process(buffer, inject: nil, collect: tap)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6

        // Intermediate taps hold densities or log exposures. Normalise them so they are viewable.
        let display = tap == .rgbOut ? rendered : Self.normaliseForDisplay(rendered, tap: tap)
        let space =
            tap == .rgbOut
            ? ImageBridge.colourSpace(forOutput: params.io.outputColourSpace)
            : CGColorSpace(name: CGColorSpace.sRGB)!
        let image = try ImageBridge.image(from: display, colourSpace: space)

        return Result(
            image: image,
            quality: quality,
            tap: tap,
            milliseconds: elapsed,
            stages: simulator.timings.sorted { $0.value > $1.value }.map {
                ($0.key, $0.value * 1000)
            },
            pixelSize: (buffer.width, buffer.height),
            wasDownscaled: quality == .full && cap != nil
        )
    }

    private func simulator(for params: RuntimePhotoParams) throws -> Simulator {
        if let simulator, simulatorParams == params { return simulator }
        // Auto-exposure meters on a downsampled preview, so the resampler is not optional. The Metal
        // backend meets the same parity tolerance as the CPU, and falls back to it without a GPU.
        let built = try Simulator(params, resampler: SkimageResampler(), backend: .metal)
        simulator = built
        simulatorParams = params
        return built
    }

    /// Scales a density or log-exposure tap to the observed range, for display.
    ///
    /// Densities run roughly 0 to 3 and log exposures go negative. Density is inverted, so dense
    /// areas show dark, as on a real negative. The taps hold density above base, so the orange mask
    /// of a colour negative does not appear.
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
