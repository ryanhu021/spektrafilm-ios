import Foundation
import os

/// Where the heavy per-pixel operators run.
public enum ComputeBackend: Sendable, Equatable {
    /// float64 on the CPU. The parity-tested reference, and the default.
    case cpu
    /// float32 on the GPU for the operators that have a Metal version, the CPU for the rest. Uses
    /// the CPU throughout on a device without Metal.
    case metal
}

/// The spectral contraction on the device the backend selects.
///
/// A GPU failure falls back to the CPU for that call, so a render completes either way.
struct SpectralProjector: @unchecked Sendable {
    private static let log = Logger(subsystem: "dev.ryanhu.spektrafilm", category: "metal")

    #if canImport(Metal)
    private let metal: MetalContext?
    #endif

    init(_ backend: ComputeBackend) {
        #if canImport(Metal)
        metal = backend == .metal ? MetalContext.shared : nil
        #endif
    }

    func project(
        cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        response: [Double],
        scale: Double = 1.0
    ) -> ImageBuffer {
        #if canImport(Metal)
        if let metal {
            do {
                return try MetalSpectralContraction.project(
                    metal, cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity,
                    illuminant: illuminant, response: response, scale: scale)
            } catch {
                Self.log.error("Metal spectral contraction failed, using the CPU: \(error)")
            }
        }
        #endif
        return SpectralContraction.project(
            cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity,
            illuminant: illuminant, response: response, scale: scale)
    }
}
