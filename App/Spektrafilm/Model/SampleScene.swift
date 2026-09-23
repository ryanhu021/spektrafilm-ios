import CoreGraphics
import Foundation

/// A synthetic scene, for launching into a loaded state without a photo library.
///
/// Generated rather than bundled: it costs no app size, it is identical on every run so two
/// screenshots are comparable, and it is built to exercise the parts of the render that a snapshot of
/// someone's holiday would not. The top half is a set of saturated patches that land outside the
/// output gamut, which is what the gamut compression is for; the bottom is a twelve-stop luminance
/// ramp, which shows the film's toe and shoulder.
///
/// Reached with `-sample` as a launch argument, so it never appears for a user.
enum SampleScene {
    static var isRequested: Bool {
        ProcessInfo.processInfo.arguments.contains("-sample")
    }

    /// Scene-linear values, deliberately including some above 1.0 so the highlight roll-off is
    /// visible. The engine treats its input as scene light, not as display values.
    static func make(width: Int = 900, height: Int = 1200) -> CGImage? {
        let patches: [(Double, Double, Double)] = [
            (0.85, 0.12, 0.10), (0.92, 0.45, 0.06), (0.88, 0.78, 0.10),
            (0.14, 0.62, 0.22), (0.08, 0.38, 0.75), (0.42, 0.14, 0.62),
            (0.95, 0.72, 0.60), (0.55, 0.36, 0.26), (0.18, 0.18, 0.20),
            (2.40, 2.10, 1.80), (0.03, 0.03, 0.035), (0.184, 0.184, 0.184),
        ]

        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        let patchRows = 3
        let patchColumns = 4
        let patchZone = Int(Double(height) * 0.58)

        for y in 0..<height {
            for x in 0..<width {
                var rgb: (Double, Double, Double)
                if y < patchZone {
                    let row = min(y * patchRows / patchZone, patchRows - 1)
                    let column = min(x * patchColumns / width, patchColumns - 1)
                    rgb = patches[row * patchColumns + column]
                } else {
                    // Twelve stops, dark to bright, spanning past 1.0 at the top end.
                    let t = Double(x) / Double(width - 1)
                    let stops = -8.0 + t * 12.0
                    let v = 0.184 * Foundation.pow(2.0, stops)
                    // Three bands: neutral, then warm and cool, so the colour balance is legible.
                    let band = (y - patchZone) * 3 / max(1, height - patchZone)
                    switch band {
                    case 0: rgb = (v, v, v)
                    case 1: rgb = (v * 1.25, v * 0.95, v * 0.72)
                    default: rgb = (v * 0.74, v * 0.95, v * 1.28)
                    }
                }

                // Encode with the sRGB transfer function, since the image is tagged sRGB and the
                // bridge relies on CoreGraphics to linearise it on import.
                func encode(_ linear: Double) -> UInt8 {
                    let clamped = min(max(linear, 0), 1)
                    let encoded =
                        clamped <= 0.0031308
                        ? clamped * 12.92
                        : 1.055 * Foundation.pow(clamped, 1.0 / 2.4) - 0.055
                    return UInt8((min(max(encoded, 0), 1) * 255).rounded())
                }

                let offset = (y * width + x) * 4
                bytes[offset] = encode(rgb.0)
                bytes[offset + 1] = encode(rgb.1)
                bytes[offset + 2] = encode(rgb.2)
            }
        }

        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
            let space = CGColorSpace(name: CGColorSpace.sRGB)
        else { return nil }

        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
