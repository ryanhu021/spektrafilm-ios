import Foundation

/// `utils/fast_gaussian_filter.py`'s `fast_exponential_filter`.
///
/// Approximates convolution with the isotropic 2D exponential `exp(-r / lambda) / (2 pi lambda^2)`
/// as a fixed sum of separable Gaussians. Each component runs the full per-channel dispatch, so a
/// three-component mixture makes three independent FIR-or-IIR decisions per channel.
public enum ExponentialFilter {

    /// How many Gaussians stand in for the exponential. Upstream only has fits for two and three.
    public enum MixtureSize: Int, Sendable, CaseIterable {
        case two = 2
        case three = 3
    }

    /// `_EXPONENTIAL_GAUSSIAN_FITS`, as `(amplitude, sigma / lambda)` in table order.
    ///
    /// The three-component amplitudes sum to 0.9999, not 1. Upstream calls these placeholder fits
    /// and never renormalises them, so the mixture loses a hundredth of a percent of the energy it
    /// claims to preserve. Renormalising here would move every scatter tail off the goldens.
    public static func fit(_ size: MixtureSize) -> [(amplitude: Double, sigmaRatio: Double)] {
        switch size {
        case .two:
            return [(0.6235, 0.9401), (0.3765, 2.5177)]
        case .three:
            return [(0.1633, 0.5360), (0.6496, 1.5236), (0.1870, 2.7684)]
        }
    }

    /// One decay constant for every channel, in pixels.
    public static func apply(
        _ image: ImageBuffer, decay: Double, mixture: MixtureSize = .three,
        truncate: Double = GaussianFilter.defaultTruncate
    ) -> ImageBuffer {
        apply(
            image,
            decayPerChannel: [Double](repeating: decay, count: image.channels),
            mixture: mixture,
            truncate: truncate
        )
    }

    /// One decay constant per channel, in pixels.
    ///
    /// Accumulates in table order. The order only matters at roundoff, and keeping it costs nothing.
    public static func apply(
        _ image: ImageBuffer, decayPerChannel: [Double], mixture: MixtureSize = .three,
        truncate: Double = GaussianFilter.defaultTruncate
    ) -> ImageBuffer {
        precondition(
            decayPerChannel.count == image.channels,
            "decay length \(decayPerChannel.count) does not match channel count \(image.channels)"
        )
        var result = ImageBuffer(
            height: image.height, width: image.width, channels: image.channels)
        for (amplitude, sigmaRatio) in fit(mixture) {
            let component = GaussianFilter.apply(
                image,
                sigmaPerChannel: decayPerChannel.map { sigmaRatio * $0 },
                truncate: truncate
            )
            result.values.withUnsafeMutableBufferPointer { dst in
                component.values.withUnsafeBufferPointer { src in
                    for i in 0..<dst.count { dst[i] += amplitude * src[i] }
                }
            }
        }
        return result
    }

    /// The same mixture on a single plane.
    public static func filterPlane(
        _ plane: [Double], height: Int, width: Int, decay: Double, mixture: MixtureSize = .three,
        truncate: Double = GaussianFilter.defaultTruncate
    ) -> [Double] {
        var result = [Double](repeating: 0, count: plane.count)
        for (amplitude, sigmaRatio) in fit(mixture) {
            let component = GaussianFilter.filterPlane(
                plane, height: height, width: width, sigma: sigmaRatio * decay, truncate: truncate)
            for i in 0..<result.count { result[i] += amplitude * component[i] }
        }
        return result
    }
}

/// The real ``SpatialFilter``: upstream's `fast_gaussian_filter` and `fast_exponential_filter`.
///
/// `model/couplers.py` calls both with their default arguments, so the defaults are the contract
/// here. ``NoSpatialFilter`` stays the right choice only when `debug.lutMode` or
/// `debug.deactivateSpatialEffects` has already zeroed the kernel sizes.
public struct FastSpatialFilter: SpatialFilter {
    public init() {}

    public func gaussian(_ image: ImageBuffer, sigma: Double) -> ImageBuffer {
        GaussianFilter.apply(image, sigma: sigma)
    }

    public func exponential(_ image: ImageBuffer, decay: Double) -> ImageBuffer {
        ExponentialFilter.apply(image, decay: decay)
    }
}
