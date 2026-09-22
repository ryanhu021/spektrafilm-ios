import Foundation

/// 1-D linear interpolation, in the two flavours the reference engine relies on.
///
/// The reference uses two different interpolators and they are *not* interchangeable:
///
/// - ``npInterp(query:xp:fp:)`` reproduces `numpy.interp`, including its guess-threaded binary
///   search. The DIR-coupler stage calls it on an axis that is not sorted for positive (slide)
///   stocks, where the search path — not just the interval — determines the answer.
/// - ``fastInterp(_:axis:values:)`` reproduces upstream's Numba kernel `fast_interp`, which
///   clamps to the endpoints and does a fresh `searchsorted` per sample. That one is
///   order-independent and is what the per-pixel density lookups use.
public enum Interpolation {

    // MARK: - numpy.interp

    /// Matches NumPy's `binary_search_with_guess` in `numpy/_core/src/multiarray/compiled_base.c`.
    ///
    /// Reproduced verbatim rather than replaced with a plain bisection because `xp` is not always
    /// sorted. `compute_density_curves_before_dir_couplers` builds its axis as
    /// `log_exposure - couplers_amount_curves`, which for positive stocks (Velvia, Provia,
    /// Ektachrome, Kodachrome) steps backwards by as much as 0.039. NumPy's result there is
    /// well-defined but depends on the guess carried over from the previous sample, the
    /// three-probe fast path and the cache-locality clamps. A textbook bisection returns a
    /// different interval, and slide film renders visibly differently.
    ///
    /// - Returns: `-1` below the range, `len` above it, otherwise an interval start index.
    @usableFromInline
    static func binarySearchWithGuess(
        key: Double, in arr: UnsafePointer<Double>, count len: Int, guess guessIn: Int
    ) -> Int {
        let likelyInCacheSize = 8

        if key > arr[len - 1] { return len }
        if key < arr[0] { return -1 }

        // NumPy linear-scans short arrays; the guess machinery below assumes len >= 5.
        if len <= 4 {
            var i = 1
            while i < len && key >= arr[i] { i += 1 }
            return i - 1
        }

        var guess = guessIn
        if guess > len - 3 { guess = len - 3 }
        if guess < 1 { guess = 1 }

        var imin = 0
        var imax = len

        if key < arr[guess] {
            if key < arr[guess - 1] {
                imax = guess - 1
                if guess > likelyInCacheSize && key >= arr[guess - likelyInCacheSize] {
                    imin = guess - likelyInCacheSize
                }
            } else {
                return guess - 1
            }
        } else {
            if key < arr[guess + 1] {
                return guess
            } else if key < arr[guess + 2] {
                return guess + 1
            } else {
                imin = guess + 2
                if guess < len - likelyInCacheSize - 1 && key < arr[guess + likelyInCacheSize] {
                    imax = guess + likelyInCacheSize
                }
            }
        }

        while imin < imax {
            let imid = imin + ((imax - imin) >> 1)
            if key >= arr[imid] {
                imin = imid + 1
            } else {
                imax = imid
            }
        }
        return imin - 1
    }

    /// `numpy.interp(query, xp, fp)` with default `left`/`right` (the endpoint values).
    ///
    /// `query` is processed in order, because the search guess is carried from one sample to the
    /// next exactly as NumPy does. Do not parallelise this or reorder the input.
    public static func npInterp(query: [Double], xp: [Double], fp: [Double]) -> [Double] {
        precondition(xp.count == fp.count, "xp and fp must be the same length")
        let len = xp.count
        precondition(len > 0, "xp must not be empty")

        var out = [Double](repeating: 0, count: query.count)
        if len == 1 {
            for i in query.indices { out[i] = query[i].isNaN ? query[i] : fp[0] }
            return out
        }

        let lval = fp[0]
        let rval = fp[len - 1]

        xp.withUnsafeBufferPointer { xpBuf in
            fp.withUnsafeBufferPointer { fpBuf in
                let dx = xpBuf.baseAddress!
                let dy = fpBuf.baseAddress!

                // NumPy precomputes slopes when `lenxp <= lenx`; the arithmetic is identical
                // either way, so always precompute.
                var slopes = [Double](repeating: 0, count: len - 1)
                for i in 0..<(len - 1) {
                    slopes[i] = (dy[i + 1] - dy[i]) / (dx[i + 1] - dx[i])
                }

                var j = 0
                for i in query.indices {
                    let x = query[i]
                    if x.isNaN {
                        out[i] = x
                        continue
                    }
                    j = binarySearchWithGuess(key: x, in: dx, count: len, guess: j)
                    if j == -1 {
                        out[i] = lval
                    } else if j == len {
                        out[i] = rval
                    } else if j == len - 1 {
                        out[i] = dy[j]
                    } else if dx[j] == x {
                        // NumPy short-circuits an exact hit to dodge a non-finite slope.
                        out[i] = dy[j]
                    } else {
                        let slope = slopes[j]
                        var res = slope * (x - dx[j]) + dy[j]
                        if res.isNaN {
                            res = slope * (x - dx[j + 1]) + dy[j + 1]
                            if res.isNaN && dy[j] == dy[j + 1] { res = dy[j] }
                        }
                        out[i] = res
                    }
                }
            }
        }
        return out
    }

    // MARK: - fast_interp

    /// Index of the first element strictly greater than `key`, i.e. `searchsorted(side: .right)`.
    @inlinable
    static func upperBound(_ key: Double, _ arr: UnsafePointer<Double>, stride: Int, count: Int)
        -> Int
    {
        var lo = 0
        var hi = count
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1)
            if arr[mid * stride] <= key { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// Upstream's `fast_interp`: per-channel 1-D interpolation of a 3-channel buffer, clamped to
    /// the endpoint values outside the axis range.
    ///
    /// - Parameters:
    ///   - image: 3-channel buffer of x values.
    ///   - axis: ascending x axis. Either `count` values shared by all channels, or `count * 3`
    ///     values interleaved per channel (upstream's `(K, 3)` case, used to fold the per-channel
    ///     `gamma_factor` into the axis).
    ///   - values: `count * 3` interleaved y values.
    ///
    /// Repeated x values are allowed: upstream stores a reciprocal of 0 for a zero-width interval,
    /// which makes the interpolation weight 0 and returns the lower y. That is reproduced here,
    /// rather than dividing and getting an infinity.
    public static func fastInterp(
        _ image: ImageBuffer, axis: [Double], values: [Double]
    ) -> ImageBuffer {
        precondition(image.channels == 3, "fastInterp expects a 3-channel buffer")
        // A per-channel axis is interleaved and therefore exactly as long as `values`.
        let perChannelAxis = axis.count == values.count
        let count = perChannelAxis ? values.count / 3 : axis.count
        precondition(values.count == count * 3, "values must be count * 3")
        precondition(count >= 2, "axis needs at least two samples")

        var out = ImageBuffer(height: image.height, width: image.width, channels: 3)

        // Reciprocal interval widths, 0 for repeated x (upstream's guard against dividing by 0).
        let axisStride = perChannelAxis ? 3 : 1
        var invDx = [Double](repeating: 0, count: (count - 1) * (perChannelAxis ? 3 : 1))
        axis.withUnsafeBufferPointer { ax in
            let a = ax.baseAddress!
            if perChannelAxis {
                for c in 0..<3 {
                    for i in 0..<(count - 1) {
                        let d = a[(i + 1) * 3 + c] - a[i * 3 + c]
                        invDx[i * 3 + c] = d != 0 ? 1.0 / d : 0.0
                    }
                }
            } else {
                for i in 0..<(count - 1) {
                    let d = a[i + 1] - a[i]
                    invDx[i] = d != 0 ? 1.0 / d : 0.0
                }
            }
        }

        image.values.withUnsafeBufferPointer { src in
            axis.withUnsafeBufferPointer { ax in
                values.withUnsafeBufferPointer { ys in
                    invDx.withUnsafeBufferPointer { inv in
                        out.values.withUnsafeMutableBufferPointer { dst in
                            let s = src.baseAddress!
                            let a = ax.baseAddress!
                            let y = ys.baseAddress!
                            let iv = inv.baseAddress!
                            let d = dst.baseAddress!
                            let n = image.pixelCount

                            for p in 0..<n {
                                for c in 0..<3 {
                                    let x = s[p * 3 + c]
                                    let axBase = perChannelAxis ? a + c : a
                                    let invBase = perChannelAxis ? iv + c : iv
                                    let first = axBase[0]
                                    let last = axBase[(count - 1) * axisStride]
                                    if x.isNaN {
                                        // Upstream reads out of bounds here (Numba runs without
                                        // bounds checks), which is undefined rather than a
                                        // behaviour worth copying. Propagate instead.
                                        d[p * 3 + c] = .nan
                                    } else if x <= first {
                                        d[p * 3 + c] = y[c]
                                    } else if x >= last {
                                        d[p * 3 + c] = y[(count - 1) * 3 + c]
                                    } else {
                                        let idx = upperBound(
                                            x, axBase, stride: axisStride, count: count)
                                        let low = idx - 1
                                        let x0 = axBase[low * axisStride]
                                        let t = (x - x0) * invBase[low * axisStride]
                                        let y0 = y[low * 3 + c]
                                        let y1 = y[(low + 1) * 3 + c]
                                        d[p * 3 + c] = y0 + t * (y1 - y0)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
