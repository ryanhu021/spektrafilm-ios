/// Index maps that fold an out-of-range index back into `0..<count`.
///
/// Three of them, named for what they do to the edge sample, because the reference uses two
/// different folds and calls both "reflect". Swapping them changes only border pixels, and a coarse
/// tolerance hides the mistake: `apply_diffusion_filter_um` with the wrong fold is off by 6.6e-6,
/// and the parity gate is 1e-4.
///
/// The naming mapping, verified index for index against the oracle:
///
/// | helper | `numpy.pad` mode | `scipy.ndimage` mode | period |
/// |---|---|---|---|
/// | ``reflectEdgeDuplicated(_:count:)`` | `symmetric` | `reflect` | `2n` |
/// | ``mirrorEdgeShared(_:count:)`` | `reflect` | `mirror` | `2(n - 1)` |
/// | ``clampEdge(_:count:)`` | `edge` | `nearest` | none |
///
/// NumPy and `scipy.ndimage` use the word "reflect" for opposite conventions, so read the mode
/// string at a call site before picking a helper.
public enum BoundaryIndex {

    /// Half-sample symmetric fold, period `2n`, edge sample duplicated: `d c b a | a b c d | d c b a`.
    ///
    /// Upstream's `_reflect(i, n)` in `fast_gaussian_filter.py`, used by the FIR blur path.
    ///
    /// Swift needs the modulo fixup. Python's `%` returns a non-negative result for a positive
    /// modulus, so upstream's `if i < 0: i += period` never runs. Swift's `%` truncates toward
    /// zero, so without the fixup a negative index goes out of bounds. The modulo path is reached
    /// whenever `radius > n`, e.g. a one-row image with a 19-tap kernel.
    @inlinable
    public static func reflectEdgeDuplicated(_ i: Int, count n: Int) -> Int {
        precondition(n > 0, "count must be positive")
        if i >= 0 && i < n { return i }
        if i >= -n && i < 0 { return -i - 1 }
        if i >= n && i < 2 * n { return 2 * n - 1 - i }
        let period = 2 * n
        var k = i % period
        if k < 0 { k += period }
        if k >= n { k = period - 1 - k }
        return k
    }

    /// Whole-sample symmetric fold, period `2(n - 1)`, edge sample shared: `d c b | a b c d | c b a`.
    ///
    /// What `numpy.pad(mode: "reflect")` does, and therefore the boundary of
    /// `apply_diffusion_filter_um` and of the resample path. `n == 1` has no period and degenerates
    /// to the single sample, matching `numpy.pad`.
    @inlinable
    public static func mirrorEdgeShared(_ i: Int, count n: Int) -> Int {
        precondition(n > 0, "count must be positive")
        if n == 1 { return 0 }
        if i >= 0 && i < n { return i }
        let period = 2 * (n - 1)
        var k = i % period
        if k < 0 { k += period }
        if k >= n { k = period - k }
        return k
    }

    /// Edge replication: `a a a | a b c d | d d d`.
    ///
    /// The IIR blur path's boundary, where it appears as initialising the recursion state from the
    /// first or last sample.
    @inlinable
    public static func clampEdge(_ i: Int, count n: Int) -> Int {
        precondition(n > 0, "count must be positive")
        if i < 0 { return 0 }
        if i >= n { return n - 1 }
        return i
    }
}
