import Foundation

/// Resamples an image by a scale factor.
///
/// Declared as a protocol because the reference uses `skimage.transform.rescale`, whose behaviour is
/// four stages deep and has its own parity notes (the output clip uses the whole array's min and max
/// rather than per channel, which moves order-3 upscale pixels by 3.7e-4 against a 1e-4 gate). The
/// default render path never resamples: `crop` is off, `upscaleFactor` is 1, and auto-exposure only
/// downsamples when the image is larger than its preview bound.
public protocol Resampler: Sendable {
    /// `skimage.transform.rescale(image, factor, channel_axis: 2, order: order)`.
    ///
    /// `order` is the spline order: 0 is nearest neighbour, 3 is bicubic.
    func rescale(_ image: ImageBuffer, factor: Double, order: Int) throws -> ImageBuffer
}

/// Refuses to resample.
///
/// Lets the pipeline run its default path before the resampler lands, and fails loudly instead of
/// silently skipping a scale the caller asked for.
public struct UnavailableResampler: Resampler {
    public init() {}

    public func rescale(_ image: ImageBuffer, factor: Double, order: Int) throws -> ImageBuffer {
        if factor == 1.0 { return image }
        throw SpektraError.unsupportedSetting(
            "resampling", value: "factor \(factor), order \(order)")
    }
}

/// Crop and scale, and the pixel pitch every micron-specified effect depends on.
///
/// Ports `runtime/services/resize.py`. `pixelSizeMicrons` is set by ``cropAndRescale`` and read
/// afterwards by the coupler, halation and grain models, so the order matters: nothing that works in
/// microns can run before preprocessing.
public final class ResizingService {
    private let io: IOParams
    private let resampler: any Resampler

    public let filmFormatMillimetres: Double

    /// Microns per pixel, from the film format and the image's long edge. `nil` until
    /// ``cropAndRescale(_:)`` has run, which is how the reference signals a pipeline injected past
    /// preprocessing, as a LUT bake does.
    public private(set) var pixelSizeMicrons: Double?

    public init(io: IOParams, filmFormatMillimetres: Double, resampler: any Resampler) {
        self.io = io
        self.filmFormatMillimetres = filmFormatMillimetres
        self.resampler = resampler
    }

    public func cropAndRescale(_ image: ImageBuffer) throws -> ImageBuffer {
        pixelSizeMicrons = filmFormatMillimetres * 1000 / Double(max(image.height, image.width))

        var result = image
        if io.crop {
            result = try Self.crop(result, center: io.cropCenter, size: io.cropSize)
        }
        if io.upscaleFactor != 1.0 {
            pixelSizeMicrons = pixelSizeMicrons! / io.upscaleFactor
            result = try resampler.rescale(result, factor: io.upscaleFactor, order: 3)
        }
        return result
    }

    /// `small_preview`. Auto-exposure meters on this, not on the full frame.
    ///
    /// Nearest neighbour, matching the reference's `order=0`.
    public func smallPreview(_ image: ImageBuffer, maxSize: Int = 256) throws -> ImageBuffer {
        let longEdge = max(image.height, image.width)
        guard longEdge > maxSize else { return image }
        return try resampler.rescale(
            image, factor: Double(maxSize) / Double(longEdge), order: 0)
    }

    /// `utils/crop_resize.crop_image`. Centre and size are fractions of the full frame.
    static func crop(
        _ image: ImageBuffer, center: (x: Double, y: Double), size: (width: Double, height: Double)
    ) throws -> ImageBuffer {
        let height = Int((Double(image.height) * size.height).rounded())
        let width = Int((Double(image.width) * size.width).rounded())
        guard height > 0, width > 0 else {
            throw SpektraError.unsupportedSetting(
                "io.crop_size", value: "\(size.width) x \(size.height) leaves no pixels")
        }
        let top = Int((Double(image.height) * center.y - Double(height) / 2).rounded())
        let left = Int((Double(image.width) * center.x - Double(width) / 2).rounded())
        let clampedTop = min(max(0, top), image.height - height)
        let clampedLeft = min(max(0, left), image.width - width)

        var out = ImageBuffer(height: height, width: width, channels: image.channels)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<image.channels {
                    out[y, x, c] = image[clampedTop + y, clampedLeft + x, c]
                }
            }
        }
        return out
    }
}
