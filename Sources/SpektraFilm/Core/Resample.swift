import Foundation

/// `skimage.transform.rescale`, for the orders the pipeline actually asks for.
///
/// Auto-exposure meters on a 256 px preview, so this is on the default render path even though no
/// user setting mentions resampling. Two details decide whether the metered EV matches:
///
/// - Anti-aliasing is **on** for the nearest-neighbour preview. skimage enables it whenever the
///   output is smaller than the input and the input is not an integer type, which a float64 image
///   is not, so `order: 0` gets a Gaussian prefilter before the nearest pick. Skipping it shifts the
///   metered EV by up to 3e-3, which is a 0.2% gain error over the whole frame.
/// - Half rounds **up** in the order-0 pick, so a coordinate of 2.5 samples index 3.
///
/// Order 3 is not implemented. It is only reachable from `io.upscaleFactor != 1`, which needs the
/// cubic spline prefilter as well as the B-spline taps, and nothing in the app sets it.
public struct SkimageResampler: Resampler {
    public init() {}

    public func rescale(_ image: ImageBuffer, factor: Double, order: Int) throws -> ImageBuffer {
        guard factor != 1.0 else { return image }
        guard order == 0 else {
            throw SpektraError.unsupportedSetting(
                "resampling order", value: "\(order); only nearest neighbour is implemented")
        }

        // skimage rounds the scaled shape, and uses np.round, which is half-to-even.
        let outHeight = max(1, Int(halfToEven(Double(image.height) * factor)))
        let outWidth = max(1, Int(halfToEven(Double(image.width) * factor)))

        var working = image
        // Anti-alias first, per axis, skipping any axis that is not shrinking. The channel axis never
        // shrinks, so channels never mix.
        let heightFactor = Double(image.height) / Double(outHeight)
        let widthFactor = Double(image.width) / Double(outWidth)
        let antiAliasing = outHeight < image.height || outWidth < image.width
        if antiAliasing {
            let sigmaY = max(0, (heightFactor - 1) / 2)
            let sigmaX = max(0, (widthFactor - 1) / 2)
            if sigmaY > 1e-15 { working = Self.gaussianRows(working, sigma: sigmaY) }
            if sigmaX > 1e-15 { working = Self.gaussianColumns(working, sigma: sigmaX) }
        }

        return Self.zoomNearest(working, outHeight: outHeight, outWidth: outWidth)
    }

    /// `np.round`: halves go to the nearest even integer.
    private func halfToEven(_ v: Double) -> Double { v.rounded(.toNearestOrEven) }

    // MARK: - Gaussian prefilter

    /// `scipy.ndimage.gaussian_filter` taps for one axis.
    ///
    /// `radius = int(truncate * sigma + 0.5)` with the default truncate of 4, then a normalised
    /// Gaussian over `-radius ... radius`.
    static func kernel(sigma: Double, truncate: Double = 4.0) -> [Double] {
        let radius = Int(truncate * sigma + 0.5)
        var taps = (-radius...radius).map { Foundation.exp(-0.5 / (sigma * sigma) * Double($0 * $0)) }
        let total = taps.reduce(0, +)
        for i in taps.indices { taps[i] /= total }
        return taps
    }

    /// Correlates along the row axis with mirror boundaries.
    ///
    /// Mirror here is scipy's `mode: "mirror"`, which does not repeat the edge sample, so it is
    /// ``BoundaryIndex/mirrorEdgeShared(_:count:)`` and not the other reflection. Using the wrong one is a 6.6e-6 error,
    /// which would slip under the parity gate.
    static func gaussianRows(_ image: ImageBuffer, sigma: Double) -> ImageBuffer {
        let taps = kernel(sigma: sigma)
        let radius = taps.count / 2
        var out = image
        for y in 0..<image.height {
            for x in 0..<image.width {
                for c in 0..<image.channels {
                    var total = 0.0
                    for (j, tap) in taps.enumerated() {
                        let source = BoundaryIndex.mirrorEdgeShared(y + j - radius, count: image.height)
                        total += tap * image[source, x, c]
                    }
                    out[y, x, c] = total
                }
            }
        }
        return out
    }

    static func gaussianColumns(_ image: ImageBuffer, sigma: Double) -> ImageBuffer {
        let taps = kernel(sigma: sigma)
        let radius = taps.count / 2
        var out = image
        for y in 0..<image.height {
            for x in 0..<image.width {
                for c in 0..<image.channels {
                    var total = 0.0
                    for (j, tap) in taps.enumerated() {
                        let source = BoundaryIndex.mirrorEdgeShared(x + j - radius, count: image.width)
                        total += tap * image[y, source, c]
                    }
                    out[y, x, c] = total
                }
            }
        }
        return out
    }

    // MARK: - Nearest-neighbour zoom

    /// `scipy.ndimage.zoom(grid_mode: True, order: 0)`.
    ///
    /// The sample coordinate is `(o + 0.5) * scale - 0.5`, and the pick is `floor(c + 0.5)` with half
    /// rounding up, so 2.5 becomes 3.
    static func zoomNearest(_ image: ImageBuffer, outHeight: Int, outWidth: Int) -> ImageBuffer {
        var out = ImageBuffer(height: outHeight, width: outWidth, channels: image.channels)
        let scaleY = Double(image.height) / Double(outHeight)
        let scaleX = Double(image.width) / Double(outWidth)

        for y in 0..<outHeight {
            let cy = (Double(y) + 0.5) * scaleY - 0.5
            let sy = BoundaryIndex.mirrorEdgeShared(Int((cy + 0.5).rounded(.down)), count: image.height)
            for x in 0..<outWidth {
                let cx = (Double(x) + 0.5) * scaleX - 0.5
                let sx = BoundaryIndex.mirrorEdgeShared(Int((cx + 0.5).rounded(.down)), count: image.width)
                for c in 0..<image.channels {
                    out[y, x, c] = image[sy, sx, c]
                }
            }
        }
        return out
    }
}
