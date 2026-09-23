import Foundation

#if canImport(os)
import os
#endif

/// How large a frame this device can render.
///
/// The engine is float64 and holds several full-frame buffers at once, so peak memory is about
/// 125 MB per megapixel. Measured in release on an M4 Pro, one measurement per process:
///
/// | Frame | Peak footprint | Time |
/// |---|---|---|
/// | 2 MP | 250 MB | 0.56 s |
/// | 6 MP | 739 MB | 2.3 s |
/// | 12 MP | 1472 MB | 3.7 s |
///
/// iOS terminates a foreground app that crosses its jetsam limit, roughly 1.4 GB on a 6 GB device.
/// 6 MP fits with room to spare. 12 MP is at the limit, so the cap applies there.
///
/// Per-tap, at 2 MP: the decoded input is 26 MB/MP, filming.expose reaches 79, and filming.develop
/// reaches 125 and sets the peak. Printing and scanning add nothing on top.
///
/// Peak is what gets an app killed, so what matters is how many frames are alive at one instant,
/// not how many are allocated in total. Spectral upsampling, the coupler correction, grain and
/// halation each work a channel plane at a time or consume their input. Holding whole frames
/// instead keeps five alive in the upsampling (two suffice), eight in the coupler correction, the
/// whole sublayer split in grain, and the input, a copy and two blurs in halation.
public enum RenderBudget {

    /// Measured peak footprint per megapixel, in bytes.
    public static let bytesPerMegapixel = 125 * 1_048_576

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
