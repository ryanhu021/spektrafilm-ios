import Foundation

#if canImport(os)
import os
#endif

/// How large a frame this device can actually render.
///
/// The engine is float64 and holds several full-frame buffers at once, so peak memory is roughly
/// 230 MB per megapixel. Measured in release, with dead pipeline taps already freed:
///
/// | Frame | Peak footprint |
/// |---|---|
/// | 2 MP | 583 MB |
/// | 6 MP | 1457 MB |
/// | 12 MP | 2805 MB |
///
/// iOS terminates a foreground app that crosses its jetsam limit, which is roughly 1.4 GB on a 6 GB
/// device, so a 12 MP export would be killed rather than finish. Until the per-stage copies come
/// down, the size is capped from the memory the process is actually allowed, and the caller is told
/// what it got.
///
/// The cost per megapixel is dominated by the coupler correction, which allocates five full-frame
/// buffers, and by the stage-by-stage copies in expose and develop. Bringing those down is the work
/// that raises this ceiling; the number here is a measurement, not a target.
public enum RenderBudget {

    /// Measured peak footprint per megapixel, in bytes.
    public static let bytesPerMegapixel = 230 * 1_048_576

    /// Fraction of the available allowance to actually spend.
    ///
    /// The rest covers the decoded source image, the UI, and the fact that the footprint figure is a
    /// high-water mark rather than a steady state.
    public static let safetyFraction = 0.55

    /// Bytes this process may still allocate before the system intervenes.
    ///
    /// On iOS this is `os_proc_available_memory`, which reports the remaining jetsam allowance.
    /// Elsewhere it reports the physical memory, since a desktop process is not jetsammed.
    public static func availableBytes() -> Int {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        let available = os_proc_available_memory()
        if available > 0 { return available }
        #endif
        return Int(ProcessInfo.processInfo.physicalMemory / 2)
    }

    /// The largest frame worth attempting, in megapixels.
    ///
    /// Never below 0.5 MP: a device that cannot manage that cannot run the engine at all, and
    /// returning zero would leave the caller with nothing to render.
    public static func maximumMegapixels() -> Double {
        let spend = Double(availableBytes()) * safetyFraction
        return max(0.5, spend / Double(bytesPerMegapixel))
    }

    /// The long edge to render at, for a source of the given size.
    ///
    /// Returns `nil` when the source already fits, so the caller can render it untouched.
    public static func longEdge(forWidth width: Int, height: Int) -> Int? {
        let megapixels = Double(width * height) / 1e6
        let limit = maximumMegapixels()
        guard megapixels > limit else { return nil }
        let scale = (limit / megapixels).squareRoot()
        return max(1, Int((Double(max(width, height)) * scale).rounded()))
    }
}
