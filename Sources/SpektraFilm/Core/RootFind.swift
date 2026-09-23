import Foundation

/// Scalar root finding, as `scipy.optimize.brentq` does it.
///
/// One caller today: the print-curve morph solves for the horizontal shift that restores density at
/// zero log exposure after developer exhaustion skews the layer CDFs.
///
/// ``brentq(lower:upper:xTolerance:relativeTolerance:maxIterations:_:)`` is a transliteration of
/// SciPy's `Zeros/brentq.c`, so the iterates and the step-acceptance test are the reference's. On ten
/// functions with known roots it returns bit-identical results at both `xtol = 2e-12` and
/// `xtol = 1e-10`, with matching iteration counts.
///
/// Nothing here throws. Per ``SpektraError``'s contract numeric problems are reported in the return
/// value: a bracket that does not straddle a sign change is `nil` (SciPy raises `ValueError`), an
/// exhausted iteration budget is ``Root/converged`` false with the last iterate (SciPy raises
/// `RuntimeError`), and a NaN residual is `nil` at an endpoint or ``Root/converged`` false mid-solve
/// (SciPy wraps `f` and raises `ValueError` on the first NaN it sees).
public enum RootFind {

    /// SciPy's `brentq` defaults. `morph_curves` overrides `xtol` to 1e-10 and leaves these alone.
    public static let defaultXTolerance = 2e-12
    public static let defaultRelativeTolerance = 4.0 * Double.ulpOfOne
    public static let defaultMaxIterations = 100

    public struct Root: Sendable, Equatable {
        public let value: Double
        public let iterations: Int
        public let converged: Bool
    }

    /// Brent's method on `[lower, upper]`, which must straddle a sign change.
    ///
    /// - Returns: `nil` when `f(lower)` and `f(upper)` share a sign bit, or when either is NaN.
    public static func brentq(
        lower: Double,
        upper: Double,
        xTolerance: Double = defaultXTolerance,
        relativeTolerance: Double = defaultRelativeTolerance,
        maxIterations: Int = defaultMaxIterations,
        _ f: (Double) -> Double
    ) -> Root? {
        precondition(xTolerance > 0, "xTolerance must be positive, got \(xTolerance)")
        precondition(maxIterations >= 0, "maxIterations must not be negative, got \(maxIterations)")

        var xPre = lower
        var xCur = upper
        var xBlk = 0.0
        var fPre = f(xPre)
        var fCur = f(xCur)
        var fBlk = 0.0
        var sPre = 0.0
        var sCur = 0.0

        if fPre.isNaN || fCur.isNaN { return nil }
        if fPre == 0 { return Root(value: xPre, iterations: 0, converged: true) }
        if fCur == 0 { return Root(value: xCur, iterations: 0, converged: true) }
        // C's `signbit`. Two same-signed residuals near 1e-200 multiply to +0.0, so a product test
        // here would read a one-sided bracket as valid and iterate on garbage.
        if fPre.sign == fCur.sign { return nil }

        var iterations = 0
        for _ in 0..<maxIterations {
            iterations += 1
            if fPre != 0 && fCur != 0 && fPre.sign != fCur.sign {
                xBlk = xPre
                fBlk = fPre
                sPre = xCur - xPre
                sCur = sPre
            }
            if abs(fBlk) < abs(fCur) {
                (xPre, xCur, xBlk) = (xCur, xBlk, xCur)
                (fPre, fCur, fBlk) = (fCur, fBlk, fCur)
            }

            let delta = (xTolerance + relativeTolerance * abs(xCur)) / 2
            let sBis = (xBlk - xCur) / 2
            if fCur == 0 || abs(sBis) < delta {
                return Root(value: xCur, iterations: iterations, converged: true)
            }

            if abs(sPre) > delta && abs(fCur) < abs(fPre) {
                let sTry: Double
                if xPre == xBlk {
                    sTry = -fCur * (xCur - xPre) / (fCur - fPre)
                } else {
                    let dPre = (fPre - fCur) / (xPre - xCur)
                    let dBlk = (fBlk - fCur) / (xBlk - xCur)
                    sTry = -fCur * (fBlk * dBlk - fPre * dPre) / (dBlk * dPre * (fBlk - fPre))
                }
                if 2 * abs(sTry) < min(abs(sPre), 3 * abs(sBis) - delta) {
                    sPre = sCur
                    sCur = sTry
                } else {
                    sPre = sBis
                    sCur = sBis
                }
            } else {
                sPre = sBis
                sCur = sBis
            }

            xPre = xCur
            fPre = fCur
            xCur += abs(sCur) > delta ? sCur : (sBis > 0 ? delta : -delta)
            fCur = f(xCur)
            if fCur.isNaN { return Root(value: xCur, iterations: iterations, converged: false) }
        }
        return Root(value: xCur, iterations: iterations, converged: false)
    }

    /// The doubling bracket search from `morph_curves._developer_exhaustion_center_offset`.
    ///
    /// Tests 12 brackets, `±0.25` through `±512`, and runs ``brentq`` on the first one that straddles
    /// a sign change. A residual that is exactly zero at either end short-circuits to that end.
    ///
    /// The reference guards this search with two early returns that belong to the morph, so they stay
    /// at the call site: `allclose(mix, 0)` with `atol = 1e-8`, and `|residual(0)| <= 1e-12`. Both
    /// yield a zero offset without solving, and dropping either changes the shipped render.
    ///
    /// - Returns: `nil` when no bracket straddles a sign change, or when ``brentq`` does not
    ///   converge on the one that does. The morph treats either as a zero offset.
    public static func expandingBracketRoot(
        lower: Double = -0.25,
        upper: Double = 0.25,
        doublings: Int = 12,
        xTolerance: Double = 1e-10,
        relativeTolerance: Double = defaultRelativeTolerance,
        maxIterations: Int = defaultMaxIterations,
        _ f: (Double) -> Double
    ) -> Double? {
        var lo = lower
        var hi = upper
        var rLo = f(lo)
        var rHi = f(hi)
        for _ in 0..<doublings {
            if rLo == 0.0 { return lo }
            if rHi == 0.0 { return hi }
            // The reference's product test. Keep it a product: for residuals below ~1e-160 it
            // underflows to ±0.0 and rejects a bracket that a sign-bit test would accept.
            if rLo * rHi < 0.0 {
                let root = brentq(
                    lower: lo,
                    upper: hi,
                    xTolerance: xTolerance,
                    relativeTolerance: relativeTolerance,
                    maxIterations: maxIterations,
                    f
                )
                guard let root, root.converged else { return nil }
                return root.value
            }
            lo *= 2.0
            hi *= 2.0
            rLo = f(lo)
            rHi = f(hi)
        }
        return nil
    }
}
