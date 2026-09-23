import Foundation

/// `utils/fast_gaussian_filter.py`'s `fast_gaussian_filter`.
///
/// Two paths behind one sigma threshold: a separable truncated FIR below 3 pixels, and the
/// Young and van Vliet third-order recursion at 3 and above. The two disagree by about 1e-1 at the
/// crossover, and neither is an accurate Gaussian. Reproduce this behaviour; do not repair it. See
/// ``iirPlane(_:height:width:sigma:)`` for the two measured deviations and why they stay.
///
/// Dispatch is per channel, and callers that build a Gaussian mixture dispatch again per mixture
/// component, so one call can take the FIR on red and blue and the IIR on green. At a 4000 pixel
/// long edge the scatter tail's third component has sigma `(2.942, 3.069, 2.879)` and splits this
/// way.
public enum GaussianFilter {

    /// `SMALL_SIGMA_MAX`. Sigma at or above this takes the IIR path.
    public static let smallSigmaMax = 3.0

    /// The `truncate` every live caller uses. SciPy's default is 4.0, which would widen every FIR
    /// kernel and break parity.
    public static let defaultTruncate = 3.0

    // MARK: - Image entry points

    /// One sigma for every channel.
    public static func apply(
        _ image: ImageBuffer, sigma: Double, truncate: Double = defaultTruncate
    ) -> ImageBuffer {
        apply(
            image,
            sigmaPerChannel: [Double](repeating: sigma, count: image.channels),
            truncate: truncate
        )
    }

    /// One sigma per channel, mirroring `_apply_per_channel`'s array-valued `sigma`.
    public static func apply(
        _ image: ImageBuffer, sigmaPerChannel: [Double], truncate: Double = defaultTruncate
    ) -> ImageBuffer {
        precondition(
            sigmaPerChannel.count == image.channels,
            "sigma length \(sigmaPerChannel.count) does not match channel count \(image.channels)"
        )
        var out = image
        for channel in 0..<image.channels {
            let filtered = filterPlane(
                plane(of: image, channel: channel),
                height: image.height,
                width: image.width,
                sigma: sigmaPerChannel[channel],
                truncate: truncate
            )
            write(plane: filtered, into: &out, channel: channel)
        }
        return out
    }

    // MARK: - Plane entry point

    /// `_dispatch_2d`. The branch is a bare comparison: no blend, no hysteresis.
    public static func filterPlane(
        _ plane: [Double], height: Int, width: Int, sigma: Double,
        truncate: Double = defaultTruncate
    ) -> [Double] {
        precondition(plane.count == height * width, "plane is not \(height)x\(width)")
        if sigma >= smallSigmaMax {
            return iirPlane(plane, height: height, width: width, sigma: sigma)
        }
        return firPlane(plane, height: height, width: width, sigma: sigma, truncate: truncate)
    }

    // MARK: - FIR path

    /// `_gaussian_kernel_1d`.
    ///
    /// `radius` truncates toward zero, so sigma 0.032 at truncate 3 gives radius 0: a single unit
    /// tap, an exact identity. The scatter core has radius 0 at every resolution up to 2048.
    public static func kernel1D(sigma: Double, truncate: Double) -> (kernel: [Double], radius: Int) {
        precondition(sigma > 0, "kernel1D needs a positive sigma")
        let radius = Int(truncate * sigma + 0.5)
        let size = 2 * radius + 1
        var kernel = [Double](repeating: 0, count: size)
        var total = 0.0
        for i in 0..<size {
            let x = Double(i - radius) / sigma
            let value = exp(-0.5 * x * x)
            kernel[i] = value
            total += value
        }
        for i in 0..<size { kernel[i] /= total }
        return (kernel, radius)
    }

    /// `_gaussian_filter_2d_small` / `_fir_2d_fused`: separable, vertical then horizontal, with the
    /// half-sample symmetric boundary of ``BoundaryIndex/reflectEdgeDuplicated(_:count:)``.
    ///
    /// Upstream fuses the two passes into 16-row strips and splits the horizontal pass into edges
    /// and a wrap-free interior. Both only improve locality and change nothing observable. This
    /// keeps the plain two-pass form and uses only the interior split.
    static func firPlane(
        _ plane: [Double], height n: Int, width m: Int, sigma: Double, truncate: Double
    ) -> [Double] {
        if sigma <= 0 { return plane }
        let (kernel, radius) = kernel1D(sigma: sigma, truncate: truncate)
        var vertical = [Double](repeating: 0, count: n * m)
        var out = [Double](repeating: 0, count: n * m)

        plane.withUnsafeBufferPointer { src in
            kernel.withUnsafeBufferPointer { kern in
                vertical.withUnsafeMutableBufferPointer { mid in
                    let s = src.baseAddress!
                    let w = kern.baseAddress!
                    let v = mid.baseAddress!
                    Parallel.forEachChunk(of: n, cost: m * (2 * radius + 1)) { rows in
                        for i in rows {
                            let row = i * m
                            for k in -radius...radius {
                                let weight = w[k + radius]
                                let source =
                                    BoundaryIndex.reflectEdgeDuplicated(i + k, count: n) * m
                                for j in 0..<m { v[row + j] += s[source + j] * weight }
                            }
                        }
                    }
                }
                vertical.withUnsafeBufferPointer { mid in
                    out.withUnsafeMutableBufferPointer { dst in
                        let interior = 2 * radius < m
                        let v = mid.baseAddress!
                        let w = kern.baseAddress!
                        let d = dst.baseAddress!
                        Parallel.forEachChunk(of: n, cost: m * (2 * radius + 1)) { rows in
                            for i in rows {
                                let row = i * m
                                for j in 0..<m {
                                    var sum = 0.0
                                    if interior && j >= radius && j < m - radius {
                                        for k in -radius...radius {
                                            sum += v[row + j + k] * w[k + radius]
                                        }
                                    } else {
                                        for k in -radius...radius {
                                            let jj = BoundaryIndex.reflectEdgeDuplicated(
                                                j + k, count: m)
                                            sum += v[row + jj] * w[k + radius]
                                        }
                                    }
                                    d[row + j] = sum
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - IIR path

    /// `_yvv_coeffs`. The `sigma < 2.5` branch is unreachable behind the 3.0 dispatch and `q` jumps
    /// across 2.5, so do not expose this path without reading upstream first.
    public static func yvvCoefficients(
        sigma: Double
    ) -> (b: Double, b1: Double, b2: Double, b3: Double) {
        let q =
            sigma >= 2.5
            ? 0.98711 * sigma - 0.96330
            : 3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).squareRoot()
        let q2 = q * q
        let q3 = q2 * q
        let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3
        let b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3
        let b2 = -(1.4281 * q2 + 1.26661 * q3)
        let b3 = 0.422205 * q3
        return (1.0 - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0)
    }

    /// `_gaussian_filter_2d_large`: horizontal recursion into scratch, then vertical into the result.
    ///
    /// Two measured properties of this path are bugs that the reference look now depends on, so both
    /// are reproduced:
    ///
    /// - **It is 3 to 11 percent wider than the sigma asked for.** Measured on an impulse, requested
    ///   3 comes back as 3.2995 (ratio 1.0998), requested 65 as 67.761 (1.0425).
    /// - **The boundary is edge replication, not reflection**, contradicting `fast_gaussian_filter`'s
    ///   own docstring. The FIR path does reflect, so the two paths differ at the borders as well as
    ///   in width.
    ///
    /// Correcting either one moves every halation golden at sigma 3 and above by about 1e-1, a
    /// thousand times the 1e-4 parity gate.
    static func iirPlane(_ plane: [Double], height n: Int, width m: Int, sigma: Double) -> [Double] {
        if sigma <= 0 { return plane }
        // Upstream's guard against unstable coefficients. Unreachable behind the 3.0 dispatch.
        if sigma < 0.5 {
            return firPlane(plane, height: n, width: m, sigma: sigma, truncate: defaultTruncate)
        }
        let c = yvvCoefficients(sigma: sigma)
        var horizontal = [Double](repeating: 0, count: n * m)
        var out = [Double](repeating: 0, count: n * m)
        iirHorizontal(plane, into: &horizontal, height: n, width: m, c)
        iirVertical(horizontal, into: &out, height: n, width: m, c)
        return out
    }

    /// `_iir_horizontal`. The recursion state starts at the first sample going forward and at the
    /// last output sample coming back, which is the edge replication.
    private static func iirHorizontal(
        _ input: [Double], into output: inout [Double], height n: Int, width m: Int,
        _ c: (b: Double, b1: Double, b2: Double, b3: Double)
    ) {
        input.withUnsafeBufferPointer { input in
            output.withUnsafeMutableBufferPointer { output in
                let src = input.baseAddress!
                let dst = output.baseAddress!
                Parallel.forEachChunk(of: n, cost: m * 2) { rows in
                    for i in rows {
                        let row = i * m
                        var w1 = src[row]
                        var w2 = w1
                        var w3 = w1
                        for j in 0..<m {
                            let w = c.b * src[row + j] + c.b1 * w1 + c.b2 * w2 + c.b3 * w3
                            dst[row + j] = w
                            w3 = w2
                            w2 = w1
                            w1 = w
                        }
                        var y1 = dst[row + m - 1]
                        var y2 = y1
                        var y3 = y1
                        for j in stride(from: m - 1, through: 0, by: -1) {
                            let y = c.b * dst[row + j] + c.b1 * y1 + c.b2 * y2 + c.b3 * y3
                            dst[row + j] = y
                            y3 = y2
                            y2 = y1
                            y1 = y
                        }
                    }
                }
            }
        }
    }

    /// `_iir_vertical`. Same recursion down and up the columns, carrying one state triple per column
    /// so the inner loop walks a row contiguously. Columns are independent, so each chunk takes a
    /// range of them through both passes.
    private static func iirVertical(
        _ input: [Double], into output: inout [Double], height n: Int, width m: Int,
        _ c: (b: Double, b1: Double, b2: Double, b3: Double)
    ) {
        var state = [Double](repeating: 0, count: m * 3)
        input.withUnsafeBufferPointer { input in
            output.withUnsafeMutableBufferPointer { output in
                state.withUnsafeMutableBufferPointer { state in
                    let src = input.baseAddress!
                    let dst = output.baseAddress!
                    let a = state.baseAddress!
                    let b = a + m
                    let d = a + 2 * m
                    Parallel.forEachChunk(of: m, cost: n * 2) { columns in
                        for j in columns {
                            a[j] = src[j]
                            b[j] = src[j]
                            d[j] = src[j]
                        }
                        for i in 0..<n {
                            let row = i * m
                            for j in columns {
                                let w =
                                    c.b * src[row + j] + c.b1 * a[j] + c.b2 * b[j] + c.b3 * d[j]
                                dst[row + j] = w
                                d[j] = b[j]
                                b[j] = a[j]
                                a[j] = w
                            }
                        }
                        let last = (n - 1) * m
                        for j in columns {
                            a[j] = dst[last + j]
                            b[j] = dst[last + j]
                            d[j] = dst[last + j]
                        }
                        for i in stride(from: n - 1, through: 0, by: -1) {
                            let row = i * m
                            for j in columns {
                                let y =
                                    c.b * dst[row + j] + c.b1 * a[j] + c.b2 * b[j] + c.b3 * d[j]
                                dst[row + j] = y
                                d[j] = b[j]
                                b[j] = a[j]
                                a[j] = y
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Channel plumbing

    /// One channel of an interleaved buffer as a contiguous plane.
    static func plane(of image: ImageBuffer, channel: Int) -> [Double] {
        if image.channels == 1 { return image.values }
        var out = [Double](repeating: 0, count: image.pixelCount)
        let stride = image.channels
        image.values.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for p in 0..<dst.count { dst[p] = src[p * stride + channel] }
            }
        }
        return out
    }

    static func write(plane: [Double], into image: inout ImageBuffer, channel: Int) {
        let stride = image.channels
        if stride == 1 {
            image.values = plane
            return
        }
        plane.withUnsafeBufferPointer { src in
            image.values.withUnsafeMutableBufferPointer { dst in
                for p in 0..<src.count { dst[p * stride + channel] = src[p] }
            }
        }
    }
}
