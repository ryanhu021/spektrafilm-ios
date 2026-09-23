import Foundation

/// Metering, in the style of a camera's exposure meter.
///
/// Ports `utils/autoexposure.py`. Every method reduces the frame's luminance to one number, divides
/// by 18% grey, and returns the compensation in stops that would put that number at grey.
///
/// The reference meters on a 256 px preview, not the full frame, and applies the resulting gain to
/// the full frame. The pipeline keeps that split.
public enum AutoExposure {

    /// Midgray. The same 0.184 the print balance and the density references use.
    public static let midgray = 0.184

    /// `measure_autoexposure_ev`.
    ///
    /// Returns 0 EV when the measured exposure is zero or infinite, matching the reference's guard
    /// against a fully black frame.
    public static func measureEV(
        _ image: ImageBuffer,
        colourSpace: ColourSpace,
        applyCCTFDecoding: Bool,
        method: CameraParams.AutoExposureMethod
    ) -> Double {
        let luminance = luminanceY(
            image, colourSpace: colourSpace, applyCCTFDecoding: applyCCTFDecoding)
        let exposure = measureExposure(luminance, height: image.height, width: image.width, method: method)
        let ev = -Foundation.log2(exposure)
        return ev.isFinite ? ev : 0.0
    }

    /// The Y channel of XYZ.
    ///
    /// Matches `colour.RGB_to_XYZ(image, color_space, apply_cctf_decoding:)`, which passes no
    /// illuminant. colour-science then adapts from the colourspace whitepoint to itself, a product
    /// that is only near identity, so the whitepoint is passed explicitly here to reproduce it.
    static func luminanceY(
        _ image: ImageBuffer, colourSpace: ColourSpace, applyCCTFDecoding: Bool
    ) -> [Double] {
        var buffer = image
        if applyCCTFDecoding { colourSpace.transfer.decode(&buffer) }
        Colour.RGBToXYZ(&buffer, colourspace: colourSpace, illuminant: colourSpace.whitepoint)
        return (0..<buffer.pixelCount).map { buffer.values[$0 * 3 + 1] }
    }

    static func measureExposure(
        _ y: [Double], height: Int, width: Int, method: CameraParams.AutoExposureMethod
    ) -> Double {
        switch method {
        case .average:
            return mean(y) / midgray

        case .median:
            return median(y) / midgray

        case .centerWeighted:
            // Gaussian falloff from the centre, sigma 0.2 of the long edge.
            let (xs, ys) = normalizedCoordinates(height: height, width: width)
            let sigma = 0.2
            var mask = [Double](repeating: 0, count: height * width)
            var total = 0.0
            for r in 0..<height {
                for c in 0..<width {
                    let w = Foundation.exp(
                        -(xs[c] * xs[c] + ys[r] * ys[r]) / (2 * sigma * sigma))
                    mask[r * width + c] = w
                    total += w
                }
            }
            var sum = 0.0
            for i in 0..<y.count { sum += y[i] * (mask[i] / total) }
            return sum / midgray

        case .partial:
            // Hard circle at 15% radius, as Canon's partial metering does.
            let (xs, ys) = normalizedCoordinates(height: height, width: width)
            var selected: [Double] = []
            for r in 0..<height {
                for c in 0..<width {
                    if (xs[c] * xs[c] + ys[r] * ys[r]).squareRoot() < 0.15 {
                        selected.append(y[r * width + c])
                    }
                }
            }
            return mean(selected.isEmpty ? y : selected) / midgray

        case .matrix:
            // A 5x5 grid, each cell weighted by a raised cosine of its distance from the centre.
            let rows = 5
            let columns = 5
            let cellHeight = height / rows
            let cellWidth = width / columns
            var means: [Double] = []
            var weights: [Double] = []
            for r in 0..<rows {
                for c in 0..<columns {
                    var cell: [Double] = []
                    for yy in (r * cellHeight)..<min((r + 1) * cellHeight, height) {
                        for xx in (c * cellWidth)..<min((c + 1) * cellWidth, width) {
                            cell.append(y[yy * width + xx])
                        }
                    }
                    if cell.isEmpty { continue }
                    means.append(mean(cell))
                    let dy = (Double(r) - Double(rows - 1) / 2) / (Double(rows - 1) / 2)
                    let dx = (Double(c) - Double(columns - 1) / 2) / (Double(columns - 1) / 2)
                    let distance = (dx * dx + dy * dy).squareRoot() / 2.0.squareRoot()
                    weights.append(0.5 * (1.0 + Foundation.cos(Double.pi * distance)))
                }
            }
            let weightTotal = weights.reduce(0, +)
            var accumulated = 0.0
            for i in weights.indices { accumulated += (weights[i] / weightTotal) * means[i] }
            return accumulated / midgray

        case .multiZone:
            // Three concentric rings weighted 50 / 30 / 20, mimicking a multi-pattern meter.
            let (xs, ys) = normalizedCoordinates(height: height, width: width)
            let bounds = [(0.00, 0.05), (0.05, 0.25), (0.25, 0.50)]
            let ringWeights = [0.50, 0.30, 0.20]
            var weightedSum = 0.0
            var weightTotal = 0.0
            for (ring, weight) in zip(bounds, ringWeights) {
                var values: [Double] = []
                for r in 0..<height {
                    for c in 0..<width {
                        let radius = (xs[c] * xs[c] + ys[r] * ys[r]).squareRoot()
                        if radius >= ring.0 && radius < ring.1 { values.append(y[r * width + c]) }
                    }
                }
                if values.isEmpty { continue }
                weightedSum += weight * mean(values)
                weightTotal += weight
            }
            let value = weightTotal > 0 ? weightedSum / weightTotal : mean(y)
            return value / midgray

        case .highlightWeighted:
            // Y squared as the weight, so highlights dominate and film highlights survive.
            var weights = y.map { $0 * $0 }
            var total = weights.reduce(0, +)
            if total < 1e-12 {
                weights = [Double](repeating: 1, count: y.count)
                total = Double(y.count)
            }
            var sum = 0.0
            for i in y.indices { sum += y[i] * weights[i] }
            return (sum / total) / midgray
        }
    }

    /// `_normalized_coords`: coordinates with the long edge spanning [-0.5, 0.5].
    static func normalizedCoordinates(height: Int, width: Int) -> (x: [Double], y: [Double]) {
        let longEdge = Double(max(height, width))
        let normHeight = Double(height) / longEdge
        let normWidth = Double(width) / longEdge
        let xs = (0..<width).map { (Double($0) / Double(width) - 0.5) * normWidth }
        let ys = (0..<height).map { (Double($0) / Double(height) - 0.5) * normHeight }
        return (xs, ys)
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }

    /// `np.median`: the mean of the two middle elements for an even count.
    private static func median(_ values: [Double]) -> Double {
        if values.isEmpty { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[middle] }
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
}
