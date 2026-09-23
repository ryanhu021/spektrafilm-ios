import Foundation

#if canImport(os)
import os
#endif

/// How large a frame this device can render.
///
/// iOS terminates a foreground app that crosses its jetsam limit, roughly 1.4 GB on a 6 GB device.
/// Measured in release on an M4 Pro, one measurement per process:
///
/// | Frame | CPU backend | Metal, float32 path |
/// |---|---|---|
/// | 2 MP | 250 MB | 227 MB |
/// | 6 MP | 739 MB | 480 MB |
/// | 12 MP | 1472 MB | 914 MB |
///
/// The CPU backend is float64 and peaks at about 125 MB per megapixel, set by film development.
/// The Metal float32 path peaks at a fixed part for the driver plus a per-megapixel part for its
/// float32 frames, and fits roughly twice the frame in the same allowance.
///
/// Peak is what gets an app killed, so what matters is how many frames are alive at one instant,
/// not how many are allocated in total. On the CPU, spectral upsampling, the coupler correction,
/// grain and halation each work a channel plane at a time or consume their input. Holding whole
/// frames instead keeps five alive in the upsampling (two suffice), eight in the coupler
/// correction, the whole sublayer split in grain, and the input, a copy and two blurs in halation.
public enum RenderBudget {

    /// Measured peak footprint per megapixel, in bytes, on the CPU backend.
    public static let bytesPerMegapixel = 125 * 1_048_576

    /// The Metal backend's float32 path, ``Simulator/processFloat(height:width:fill:read:)``: a
    /// fixed part for the driver and staging, and a per-megapixel part for the float32 frames.
    /// Measured at 227 MB for 2 MP, 480 MB for 6 MP and 914 MB for 12 MP, which fits 90 MB plus
    /// 69 MB per megapixel. Both are rounded up here.
    public static let metalFixedBytes = 96 * 1_048_576
    public static let metalBytesPerMegapixel = 72 * 1_048_576

    /// Fraction of the available allowance to spend.
    ///
    /// The rest covers the decoded source image, the UI, and the fact that the footprint figure is a
    /// high-water mark rather than a steady state.
    public static let safetyFraction = 0.55

    /// Bytes this process may still allocate before the system intervenes.
    ///
    /// On iOS this is `os_proc_available_memory`, which reports the remaining jetsam allowance.
    /// Elsewhere it reports half the physical memory, since a desktop process is not jetsammed.
    public static func availableBytes() -> Int {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        let available = os_proc_available_memory()
        if available > 0 { return available }
        #endif
        return Int(ProcessInfo.processInfo.physicalMemory / 2)
    }

    /// The largest frame worth attempting, in megapixels, on the CPU or, with `gpu`, on the
    /// Metal backend's float32 path.
    ///
    /// Never below 0.5 MP: a device that cannot manage that cannot run the engine at all, and
    /// returning zero would leave the caller with nothing to render.
    public static func maximumMegapixels(gpu: Bool = false) -> Double {
        let spend = Double(availableBytes()) * safetyFraction
        let megapixels =
            gpu
            ? (spend - Double(metalFixedBytes)) / Double(metalBytesPerMegapixel)
            : spend / Double(bytesPerMegapixel)
        return max(0.5, megapixels)
    }

    /// The long edge to render at, for a source of the given size.
    ///
    /// Returns `nil` when the source already fits, so the caller can render it untouched.
    public static func longEdge(forWidth width: Int, height: Int, gpu: Bool = false) -> Int? {
        let megapixels = Double(width * height) / 1e6
        let limit = maximumMegapixels(gpu: gpu)
        guard megapixels > limit else { return nil }
        let scale = (limit / megapixels).squareRoot()
        return max(1, Int((Double(max(width, height)) * scale).rounded()))
    }
}
