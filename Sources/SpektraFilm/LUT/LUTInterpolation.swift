import Foundation

/// Samples a square 2D LUT at per-pixel coordinates.
///
/// A Metal implementation plugs in here. The `tc_lut` fetch is the only per-pixel work in spectral
/// upsampling, so it is the only part of the subsystem worth a GPU path. A GPU version must
/// reproduce ``MitchellLUT2DSampler`` exactly, including the non-interpolating kernel. An
/// `MTLSampler` bicubic is a different filter and shifts every pixel.
public protocol LUT2DSampler: Sendable {
    /// Samples one pixel at a time, so a caller that derives its coordinates from another buffer
    /// never has to materialise them as a frame.
    ///
    /// `coordinate` is called concurrently for different pixels, so it must only read.
    ///
    /// - Parameters:
    ///   - lut: `[gridX][gridY][channel]`, square in its two grid axes.
    ///   - destination: `[height][width][lut.channels]`, overwritten.
    ///   - coordinate: for one pixel index, the grid coordinates normalised to `[0, 1]` and a gain
    ///     applied to every fetched channel.
    func sample(
        lut: ImageBuffer,
        into destination: inout ImageBuffer,
        coordinate: (Int) -> (x: Double, y: Double, gain: Double)
    )
}

extension LUT2DSampler {
    /// - Parameters:
    ///   - lut: `[gridX][gridY][channel]`, square in its two grid axes.
    ///   - coordinates: `[height][width][>= 2]`. Channel 0 is the x coordinate, channel 1 the y,
    ///     both normalised to `[0, 1]`. Further channels are ignored.
    /// - Returns: `[height][width][lut.channels]`.
    public func sample(lut: ImageBuffer, coordinates: ImageBuffer) -> ImageBuffer {
        precondition(coordinates.channels >= 2, "LUT coordinates need at least 2 channels")
        let inChannels = coordinates.channels
        var out = ImageBuffer(
            height: coordinates.height, width: coordinates.width, channels: lut.channels)
        coordinates.values.withUnsafeBufferPointer { source in
            sample(lut: lut, into: &out) { pixel in
                (source[pixel * inChannels], source[pixel * inChannels + 1], 1.0)
            }
        }
        return out
    }
}

/// `fast_interp_lut.apply_lut_cubic_2d`: a Mitchell-Netravali fetch with B = C = 1/3.
public struct MitchellLUT2DSampler: LUT2DSampler {
    public init() {}

    public func sample(
        lut: ImageBuffer,
        into destination: inout ImageBuffer,
        coordinate: (Int) -> (x: Double, y: Double, gain: Double)
    ) {
        LUTInterpolation.sampleCubic2D(lut: lut, into: &destination, coordinate: coordinate)
    }
}

/// The 2D LUT interpolators from `utils/fast_interp_lut.py`.
///
/// LUT axis convention throughout: an ``ImageBuffer`` whose `height` axis is the LUT's first grid
/// axis, `width` its second, `channels` its output channels. For the spectral `tc_lut` that is
/// `[tc.x][tc.y][rgb]`, 192 x 192 x 3.
public enum LUTInterpolation {

    // MARK: - Kernel

    /// Mitchell-Netravali with `B = C = 1/3`, the reference's `mitchell_weight` defaults.
    ///
    /// **This kernel does not interpolate.** At `t = 0` the four weights are
    /// `[1/18, 8/9, 1/18, 0]`, so a fetch at an exact grid coordinate returns a smoothed value, not
    /// the stored one: measured 0.727 absolute worst case on the shipped irradiance LUT. The
    /// smoothing is part of the look. Substituting bilinear, Catmull-Rom or a B-spline moves every
    /// rendered pixel by percent-level amounts.
    @inlinable
    public static func mitchellWeight(_ t: Double) -> Double {
        let b = 1.0 / 3.0
        let c = 1.0 / 3.0
        let x = abs(t)
        let x2 = x * x
        let x3 = x2 * x
        if x < 1 {
            return (1.0 / 6.0) * ((12 - 9 * b - 6 * c) * x3 + (-18 + 12 * b + 6 * c) * x2 + (6 - 2 * b))
        }
        if x < 2 {
            return (1.0 / 6.0)
                * ((-b - 6 * c) * x3 + (6 * b + 30 * c) * x2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c))
        }
        return 0
    }

    // MARK: - Coordinates

    /// `clamp_coordinate`: into `[0, size - 1]`. NaN passes through, as it does in the reference,
    /// because neither comparison holds.
    @inlinable
    public static func clampCoordinate(_ coordinate: Double, size: Int) -> Double {
        if coordinate <= 0 { return 0 }
        let upper = Double(size - 1)
        if coordinate >= upper { return upper }
        return coordinate
    }

    /// `cubic_coordinate_base_fraction`: the cell index and the position inside it.
    ///
    /// The top edge maps to `(size - 2, 1.0)` instead of `(size - 1, 0.0)`, so the evaluator's
    /// `base + 1` stays in range.
    ///
    /// **NaN guard.** `Int(Double.nan)` traps in Swift, so NaN cannot reach the `Int` conversion. It
    /// maps to base 0 with a **NaN fraction**, which reproduces the reference. There,
    /// `int(np.floor(nan))` is a garbage index, but every `mitchell_weight(nan)` is 0 because both of
    /// its comparisons fail, so the weight sum is 0 and the trailing guard leaves the output at zero.
    /// A NaN fraction gives the same 16 zero weights and the same exact zero, measured against the
    /// oracle. Returning fraction 0 would fetch the smoothed cell at the origin, up to 8.1 off on
    /// the production `tc_lut`.
    @inlinable
    public static func cubicCoordinateBaseFraction(
        _ coordinate: Double, size: Int
    ) -> (base: Int, fraction: Double) {
        if coordinate.isNaN { return (0, .nan) }
        let clamped = clampCoordinate(coordinate, size: size)
        if clamped >= Double(size - 1) { return (size - 2, 1.0) }
        let base = clamped.rounded(.down)
        return (Int(base), clamped - base)
    }

    /// `safe_index`: the whole-sample mirror `scipy.ndimage` calls `mode='mirror'`.
    ///
    /// The clamp above keeps requested indices inside `[-1, size]`, so one fold always suffices.
    @inlinable
    public static func safeIndex(_ index: Int, size: Int) -> Int {
        BoundaryIndex.mirrorEdgeShared(index, count: size)
    }

    // MARK: - Point sampling

    /// `_cubic_interp_lut_at_2d`. Writes `lut.channels` values into `out`.
    ///
    /// Keep the trailing division by the weight sum: the sum is `1 ± 9e-16`, not exactly 1, so
    /// dropping it changes the output (far below the parity gate).
    @inlinable
    public static func cubicInterpLUT2D(
        lut: ImageBuffer, x: Double, y: Double, into out: inout [Double]
    ) {
        let size = lut.height
        let channels = lut.channels
        let (xBase, xFrac) = cubicCoordinateBaseFraction(x, size: size)
        let (yBase, yFrac) = cubicCoordinateBaseFraction(y, size: size)

        let wx = (
            mitchellWeight(xFrac + 1), mitchellWeight(xFrac), mitchellWeight(xFrac - 1),
            mitchellWeight(xFrac - 2)
        )
        let wy = (
            mitchellWeight(yFrac + 1), mitchellWeight(yFrac), mitchellWeight(yFrac - 1),
            mitchellWeight(yFrac - 2)
        )

        for c in 0..<channels { out[c] = 0 }
        var weightSum = 0.0
        for i in 0..<4 {
            let xi = safeIndex(xBase - 1 + i, size: size)
            let wxi = tupleElement(wx, i)
            for j in 0..<4 {
                let yj = safeIndex(yBase - 1 + j, size: size)
                let weight = wxi * tupleElement(wy, j)
                weightSum += weight
                let base = (xi * lut.width + yj) * channels
                for c in 0..<channels { out[c] += weight * lut.values[base + c] }
            }
        }
        if weightSum != 0 {
            for c in 0..<channels { out[c] /= weightSum }
        }
    }

    /// `linear_interp_lut_at_2d`: bilinear with clamp-to-edge, reached only when the LUT has fewer
    /// than two samples per axis.
    ///
    /// A NaN coordinate maps to 0 here, so the single cell is returned. The reference is undefined
    /// for that input: `int(np.floor(nan))` again, and its bilinear weights come out NaN where the
    /// cubic kernel's come out 0. The path is unreachable from the engine, because the only 2D LUT
    /// in the pipeline is 192 x 192.
    @inlinable
    public static func linearInterpLUT2D(
        lut: ImageBuffer, x: Double, y: Double, into out: inout [Double]
    ) {
        let size = lut.height
        let channels = lut.channels
        let cx = x.isNaN ? 0 : clampCoordinate(x, size: size)
        let cy = y.isNaN ? 0 : clampCoordinate(y, size: size)

        let x0 = Int(cx.rounded(.down))
        let y0 = Int(cy.rounded(.down))
        let x1 = min(x0 + 1, size - 1)
        let y1 = min(y0 + 1, size - 1)
        let tx = cx - Double(x0)
        let ty = cy - Double(y0)

        for c in 0..<channels { out[c] = 0 }
        for i in 0..<2 {
            let xi = i == 0 ? x0 : x1
            let wx = i == 0 ? 1.0 - tx : tx
            for j in 0..<2 {
                let yj = j == 0 ? y0 : y1
                let weight = wx * (j == 0 ? 1.0 - ty : ty)
                let base = (xi * lut.width + yj) * channels
                for c in 0..<channels { out[c] += weight * lut.values[base + c] }
            }
        }
    }

    // MARK: - Image application

    /// `apply_lut_cubic_2d`, including its fall-through to bilinear for a degenerate LUT.
    ///
    /// Only channels 0 and 1 of `coordinates` are read; the output carries the LUT's channel count.
    public static func applyLUTCubic2D(lut: ImageBuffer, coordinates: ImageBuffer) -> ImageBuffer {
        MitchellLUT2DSampler().sample(lut: lut, coordinates: coordinates)
    }

    /// `apply_lut_cubic_2d` with the coordinates generated per pixel and the fetch scaled per pixel,
    /// including its fall-through to bilinear for a degenerate LUT.
    ///
    /// The caller supplies `destination` and produces coordinates on demand, so neither the
    /// coordinates nor the gains are ever a frame. At 12 MP the coordinates alone are 183 MB.
    ///
    /// This is the subsystem's only per-pixel work, so the tap loop runs on raw pointers: 43 ms per
    /// megapixel with 3 output channels on an M-series core, against 58 ms for the same loop through
    /// ``ImageBuffer``'s bounds-checked subscript. The reference quotes 14.8 ms per megapixel for a
    /// Numba kernel with `parallel=True` across rows. Output pixels are independent here too, so a
    /// ``LUT2DSampler`` can parallelise across rows when a caller needs it.
    public static func sampleCubic2D(
        lut: ImageBuffer,
        into destination: inout ImageBuffer,
        coordinate: (Int) -> (x: Double, y: Double, gain: Double)
    ) {
        precondition(
            lut.height == lut.width,
            "a 2D LUT must be square; got \(lut.height)x\(lut.width)")
        precondition(
            destination.channels == lut.channels,
            "destination carries \(destination.channels) channels, LUT has \(lut.channels)")
        let size = lut.height
        let channels = lut.channels
        let scale = Double(size - 1)
        let pixelCount = destination.pixelCount

        if size < 2 {
            var pixel = [Double](repeating: 0, count: channels)
            for index in 0..<pixelCount {
                let (x, y, gain) = coordinate(index)
                linearInterpLUT2D(lut: lut, x: x * scale, y: y * scale, into: &pixel)
                for c in 0..<channels {
                    destination.values[index * channels + c] = pixel[c] * gain
                }
            }
            return
        }

        lut.values.withUnsafeBufferPointer { table in
            destination.values.withUnsafeMutableBufferPointer { destination in
                let rowStride = size * channels
                Parallel.forEachChunk(of: pixelCount, cost: 16 * channels) { pixels in
                    for index in pixels {
                        let (x, y, gain) = coordinate(index)
                        let (xBase, xFrac) = cubicCoordinateBaseFraction(x * scale, size: size)
                        let (yBase, yFrac) = cubicCoordinateBaseFraction(y * scale, size: size)
                        let wx = (
                            mitchellWeight(xFrac + 1), mitchellWeight(xFrac),
                            mitchellWeight(xFrac - 1), mitchellWeight(xFrac - 2)
                        )
                        let wy = (
                            mitchellWeight(yFrac + 1), mitchellWeight(yFrac),
                            mitchellWeight(yFrac - 1), mitchellWeight(yFrac - 2)
                        )
                        let rows = (
                            safeIndex(xBase - 1, size: size) * rowStride,
                            safeIndex(xBase, size: size) * rowStride,
                            safeIndex(xBase + 1, size: size) * rowStride,
                            safeIndex(xBase + 2, size: size) * rowStride
                        )
                        let columns = (
                            safeIndex(yBase - 1, size: size) * channels,
                            safeIndex(yBase, size: size) * channels,
                            safeIndex(yBase + 1, size: size) * channels,
                            safeIndex(yBase + 2, size: size) * channels
                        )
                        let base = index * channels
                        for c in 0..<channels { destination[base + c] = 0 }
                        var weightSum = 0.0
                        for i in 0..<4 {
                            let row = tupleElement(rows, i)
                            let weightX = tupleElement(wx, i)
                            for j in 0..<4 {
                                let weight = weightX * tupleElement(wy, j)
                                weightSum += weight
                                let cell = row + tupleElement(columns, j)
                                for c in 0..<channels {
                                    destination[base + c] += weight * table[cell + c]
                                }
                            }
                        }
                        if weightSum != 0 {
                            for c in 0..<channels { destination[base + c] /= weightSum }
                        }
                        for c in 0..<channels { destination[base + c] *= gain }

                    }
                }
            }
        }
    }

    @inlinable
    static func tupleElement(_ t: (Double, Double, Double, Double), _ index: Int) -> Double {
        switch index {
        case 0: return t.0
        case 1: return t.1
        case 2: return t.2
        default: return t.3
        }
    }

    @inlinable
    static func tupleElement(_ t: (Int, Int, Int, Int), _ index: Int) -> Int {
        switch index {
        case 0: return t.0
        case 1: return t.1
        case 2: return t.2
        default: return t.3
        }
    }
}
