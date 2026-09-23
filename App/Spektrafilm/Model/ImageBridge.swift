import CoreGraphics
import Foundation
import SpektraFilm
import UIKit

/// Converts between CoreGraphics images and the engine's buffers.
///
/// The engine wants scene-linear RGB in a named colour space. Decoding draws the image into a linear
/// CGColorSpace, so CoreGraphics applies the file's own transfer function and gamut mapping from its
/// embedded profile. The engine is then told `inputCCTFDecoding = false`, since the values are
/// already linear.
enum ImageBridge {

    /// The working colour space. Display P3 covers what modern iPhone cameras capture, and the engine
    /// registers it, so nothing is clipped on the way in.
    static let workingColourSpaceName = "Display P3"

    enum BridgeError: LocalizedError {
        case noColourSpace
        case noContext
        case noImage

        var errorDescription: String? {
            switch self {
            case .noColourSpace: return "Could not create a linear Display P3 colour space."
            case .noContext: return "Could not create a bitmap context for the photo."
            case .noImage: return "Could not read the photo."
            }
        }
    }

    /// The size `image` decodes to with its long edge at most `longEdge`.
    static func size(of image: CGImage, longEdge: Int?) -> (width: Int, height: Int) {
        var width = image.width
        var height = image.height
        if let longEdge, max(width, height) > longEdge {
            let scale = Double(longEdge) / Double(max(width, height))
            width = max(1, Int((Double(width) * scale).rounded()))
            height = max(1, Int((Double(height) * scale).rounded()))
        }
        return (width, height)
    }

    /// Decodes a `CGImage` into a linear `ImageBuffer`, scaled so its long edge is at most `longEdge`.
    ///
    /// CoreGraphics scales in the same pass as the colour conversion. The engine's resampler only
    /// implements the orders its own pipeline uses.
    static func buffer(from image: CGImage, longEdge: Int?) throws -> ImageBuffer {
        let (width, height) = size(of: image, longEdge: longEdge)
        var floats = [Float](repeating: 0, count: width * height * 3)
        try floats.withUnsafeMutableBufferPointer {
            try decode(image, width: width, height: height, into: $0)
        }
        return ImageBuffer(
            height: height, width: width, channels: 3, values: floats.map(Double.init))
    }

    /// Decodes a `CGImage` into linear float32 RGB, `height` x `width`, written to `pixels`.
    ///
    /// The source is at most 16 bits per channel, so float32 loses nothing.
    static func decode(
        _ image: CGImage, width: Int, height: Int, into pixels: UnsafeMutableBufferPointer<Float>
    ) throws {
        precondition(pixels.count == width * height * 3)
        guard let linear = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3) else {
            throw BridgeError.noColourSpace
        }
        let componentsPerPixel = 4
        let bytesPerRow = width * componentsPerPixel * 4
        var raw = [Float](repeating: 0, count: width * height * componentsPerPixel)
        let bitmapInfo: CGBitmapInfo = [
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            .floatComponents,
            .byteOrder32Little,
        ]
        try raw.withUnsafeMutableBytes { bytes in
            guard
                let context = CGContext(
                    data: bytes.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 32,
                    bytesPerRow: bytesPerRow,
                    space: linear,
                    bitmapInfo: bitmapInfo.rawValue)
            else { throw BridgeError.noContext }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        for pixel in 0..<(width * height) {
            let source = pixel * componentsPerPixel
            let alpha = raw[source + 3]
            // Un-premultiply, or a transparent PNG darkens toward black and renders as shadow.
            let scale: Float = alpha > 1e-6 ? 1 / alpha : 0
            pixels[pixel * 3] = raw[source] * scale
            pixels[pixel * 3 + 1] = raw[source + 1] * scale
            pixels[pixel * 3 + 2] = raw[source + 2] * scale
        }
    }

    /// Wraps a rendered buffer as a `CGImage` for display.
    static func image(from buffer: ImageBuffer, colourSpace: CGColorSpace) throws -> CGImage {
        precondition(buffer.channels == 3, "expected an RGB buffer")
        let floats = buffer.values.map(Float.init)
        return try floats.withUnsafeBufferPointer {
            try image(from: $0, width: buffer.width, height: buffer.height, colourSpace: colourSpace)
        }
    }

    /// An 8-bit `CGImage` from rendered float32 RGB.
    ///
    /// The engine's output is already encoded with the output colour space's transfer function, so
    /// the image is tagged with that space. Values are clamped to [0, 1]. Output gamut compression
    /// keeps them inside the cube, so anything outside is a defect, and a clip shows it plainly.
    static func image(
        from pixels: UnsafeBufferPointer<Float>, width: Int, height: Int, colourSpace: CGColorSpace
    ) throws -> CGImage {
        let count = width * height
        precondition(pixels.count == count * 3)
        var bytes = [UInt8](repeating: 255, count: count * 4)
        for pixel in 0..<count {
            for channel in 0..<3 {
                let v = pixels[pixel * 3 + channel]
                let clamped = v.isFinite ? min(max(v, 0), 1) : 0
                bytes[pixel * 4 + channel] = UInt8((clamped * 255).rounded())
            }
        }

        let provider = CGDataProvider(data: Data(bytes) as CFData)
        guard
            let provider,
            let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colourSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent)
        else { throw BridgeError.noImage }
        return image
    }

    /// The `CGColorSpace` matching an engine output colour space name.
    static func colourSpace(forOutput name: String) -> CGColorSpace {
        let fallback = CGColorSpace(name: CGColorSpace.sRGB)!
        switch name {
        case "Display P3": return CGColorSpace(name: CGColorSpace.displayP3) ?? fallback
        case "ITU-R BT.2020": return CGColorSpace(name: CGColorSpace.itur_2020) ?? fallback
        case "Adobe RGB (1998)": return CGColorSpace(name: CGColorSpace.adobeRGB1998) ?? fallback
        case "ProPhoto RGB": return CGColorSpace(name: CGColorSpace.rommrgb) ?? fallback
        default: return fallback
        }
    }
}
