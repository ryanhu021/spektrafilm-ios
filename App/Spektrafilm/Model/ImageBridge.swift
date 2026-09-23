import CoreGraphics
import Foundation
import SpektraFilm
import UIKit

/// Converts between CoreGraphics images and the engine's buffers.
///
/// The colour handling matters more than the plumbing. The engine wants scene-linear RGB in a named
/// colour space, so decoding happens here by drawing through a linear CGColorSpace: CoreGraphics
/// applies the source image's own transfer function and gamut mapping on the way in, which is more
/// trustworthy than reading encoded bytes and inverting a transfer function we guessed at. The
/// engine is then told `inputCCTFDecoding = false`, because the values arriving are already linear.
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

    /// Decodes a `CGImage` into a linear `ImageBuffer`, scaled so its long edge is at most `longEdge`.
    ///
    /// Scaling happens inside CoreGraphics rather than in the engine: the engine's own resampler is
    /// not ported, and CoreGraphics is doing a colour conversion in the same pass anyway.
    static func buffer(from image: CGImage, longEdge: Int?) throws -> ImageBuffer {
        var width = image.width
        var height = image.height
        if let longEdge, max(width, height) > longEdge {
            let scale = Double(longEdge) / Double(max(width, height))
            width = max(1, Int((Double(width) * scale).rounded()))
            height = max(1, Int((Double(height) * scale).rounded()))
        }

        guard let linear = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3) else {
            throw BridgeError.noColourSpace
        }

        // Float32 RGBA. The engine works in Double, but the source is at most 16 bits per channel, so
        // Float32 is lossless here and halves the intermediate.
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

        var values = [Double](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let source = pixel * componentsPerPixel
            let alpha = Double(raw[source + 3])
            // Un-premultiply. A transparent PNG would otherwise darken toward black, which the film
            // model would faithfully expose as a shadow.
            let scale = alpha > 1e-6 ? 1.0 / alpha : 0.0
            values[pixel * 3] = Double(raw[source]) * scale
            values[pixel * 3 + 1] = Double(raw[source + 1]) * scale
            values[pixel * 3 + 2] = Double(raw[source + 2]) * scale
        }

        return ImageBuffer(height: height, width: width, channels: 3, values: values)
    }

    /// Wraps a rendered buffer as a `CGImage` for display.
    ///
    /// The engine emits values encoded with the output colour space's transfer function, so the image
    /// is tagged with that space rather than a linear one. Values are clamped to [0, 1] here: output
    /// gamut compression already lands inside the cube, and anything outside it is a defect worth
    /// seeing as a flat clip rather than a wrapped colour.
    static func image(from buffer: ImageBuffer, colourSpace: CGColorSpace) throws -> CGImage {
        precondition(buffer.channels == 3, "expected an RGB buffer")
        let count = buffer.pixelCount
        var bytes = [UInt8](repeating: 255, count: count * 4)
        for pixel in 0..<count {
            for channel in 0..<3 {
                let v = buffer.values[pixel * 3 + channel]
                let clamped = v.isFinite ? min(max(v, 0), 1) : 0
                bytes[pixel * 4 + channel] = UInt8((clamped * 255).rounded())
            }
        }

        let provider = CGDataProvider(data: Data(bytes) as CFData)
        guard
            let provider,
            let image = CGImage(
                width: buffer.width,
                height: buffer.height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: buffer.width * 4,
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
