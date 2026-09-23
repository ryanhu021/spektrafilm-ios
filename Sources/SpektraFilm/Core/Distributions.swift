import Foundation

/// The distributions the grain and glare models draw from.
///
/// Every sampler takes its randomness from a ``RandomSource``, so a draw is fixed by the key and
/// the counter and not by how the image was decomposed.
///
/// These do not reproduce the reference's numbers. Upstream draws through SciPy on NumPy's global
/// legacy MT19937, and matching it bit for bit would mean porting `random_poisson`,
/// `random_binomial` and `random_gauss` verbatim in the same consumption order for no rendering
/// benefit (`grain.md` section 8.3). The gate is statistical instead: the closed-form moments, plus
/// a chi-square goodness of fit where the distribution is discrete enough to check.
public enum Distributions {

    // MARK: - Normal

    /// Box-Muller, taking the cosine branch and dropping the sine one.
    ///
    /// Keeping the second value would make the result depend on how many draws came before it.
    /// ``RandomSource`` exists to remove that dependence.
    @inlinable
    public static func standardNormal<R: RandomSource>(_ source: inout R) -> Double {
        let radial = source.nextOpenUniform()
        let angular = source.nextUniform()
        return (-2.0 * Foundation.log(radial)).squareRoot()
            * Foundation.cos(2.0 * Double.pi * angular)
    }

    public static func standardNormal(key: PhiloxKey, counter: UInt64) -> Double {
        var source = Philox4x32(key: key, counter: counter)
        return standardNormal(&source)
    }

    // MARK: - Lognormal

    /// Log-space parameters of the lognormal whose linear-space mean is `mean` and whose
    /// linear-space standard deviation is `std`.
    ///
    /// Transcribed from `utils/fast_stats.fast_lognormal_from_mean_std`, including its
    /// `mean <= 0` branch. Written as `log(1 + s*s/(m*m))` because that is what the reference
    /// evaluates; `log1p` is better conditioned for tiny ratios and gives different last bits.
    ///
    /// The guard is spelled as the negation of the reference's `m <= 0` so that a NaN mean falls
    /// through to the arithmetic and stays NaN. `mean > 0` would send NaN to the `mean <= 0`
    /// constants and give a clean unit field where the reference gives NaN (measured).
    @inlinable
    public static func lognormalLogParameters(
        mean: Double, std: Double
    ) -> (
        mu: Double, sigma: Double
    ) {
        guard !(mean <= 0) else { return (0.0, 0.0) }
        let sigmaSquared = Foundation.log(1.0 + (std * std) / (mean * mean))
        return (Foundation.log(mean) - sigmaSquared / 2.0, sigmaSquared.squareRoot())
    }

    /// Below this log-space sigma the reference returns `exp(mu)` and draws nothing.
    public static let lognormalSigmaFloor = 1e-6

    /// A lognormal variate with linear-space mean `mean` and standard deviation `std`.
    ///
    /// Grain's clumping field and glare's flare field are both built from this, with unit mean by
    /// construction.
    @inlinable
    public static func lognormalFromMeanStd<R: RandomSource>(
        mean: Double, std: Double, _ source: inout R
    ) -> Double {
        let (mu, sigma) = lognormalLogParameters(mean: mean, std: std)
        if sigma < lognormalSigmaFloor { return Foundation.exp(mu) }
        return Foundation.exp(mu + sigma * standardNormal(&source))
    }

    public static func lognormalFromMeanStd(
        mean: Double, std: Double, key: PhiloxKey, counter: UInt64
    ) -> Double {
        var source = Philox4x32(key: key, counter: counter)
        return lognormalFromMeanStd(mean: mean, std: std, &source)
    }

    // MARK: - Poisson

    /// Lambda at or above which ``poisson(lambda:_:)`` switches from Knuth to transformed
    /// rejection. Same crossover NumPy uses.
    public static let poissonRejectionThreshold = 10.0

    /// NumPy's `POISSON_LAM_MAX`, `Int64.max - 10 * sqrt(Int64.max)`.
    ///
    /// Above this the reference raises `ValueError: lam value too large`. ``poisson(lambda:_:)``
    /// cannot throw from inside a per-pixel loop, so it clamps instead. Without the clamp the
    /// `Int(_:)` in ``poissonTransformedRejection(lambda:_:)`` traps: `lambda = 1e19` crashes the
    /// process, measured. Grain reaches that range, because `lambda = N * p / sat` and
    /// `sat = 1 - p * u * (1 - 1e-6)` is one ulp above zero at `uniformity = 1.0000020000029999`,
    /// which puts lambda at 2.8e19 for the 6125 particles per pixel a 1000 pixel wide frame gives.
    public static let poissonLambdaMax = 9.223372006484771e18

    /// A Poisson variate.
    ///
    /// Grain needs the full span the particle model produces, roughly 4.5 to 1.2e8: `sat` falls to
    /// about 2e-6 when `uniformity` is 1 and the density saturates, and `lambda = N / sat` rises
    /// with it (`grain.md` section 4.1). One algorithm does not cover that, so this is Knuth
    /// below 10 and Hormann's transformed rejection at and above.
    ///
    /// A non-finite lambda yields 0. The reference paths give no single answer to copy: the exact
    /// path raises, `RandomState.poisson(nan)` giving `ValueError: lam < 0 or lam is NaN` and
    /// `Generator.poisson(inf)` giving `ValueError: lam value too large`, while
    /// `fast_stats.fast_poisson` returns 0 for NaN and `Int64.max` for infinity, both measured.
    /// Returning 0 keeps `Int(Double.nan)`, which traps in Swift, out of the sampler.
    @inlinable
    public static func poisson<R: RandomSource>(lambda: Double, _ source: inout R) -> Int {
        guard lambda.isFinite, lambda > 0 else { return 0 }
        if lambda < poissonRejectionThreshold {
            return poissonKnuth(lambda: lambda, &source)
        }
        return poissonTransformedRejection(
            lambda: Swift.min(lambda, poissonLambdaMax), &source)
    }

    public static func poisson(lambda: Double, key: PhiloxKey, counter: UInt64) -> Int {
        var source = Philox4x32(key: key, counter: counter)
        return poisson(lambda: lambda, &source)
    }

    /// Multiply uniforms until the product drops to `exp(-lambda)`. Exact, and it costs about
    /// `lambda + 1` uniforms, so it is only worth using while lambda is small.
    @usableFromInline
    static func poissonKnuth<R: RandomSource>(lambda: Double, _ source: inout R) -> Int {
        let threshold = Foundation.exp(-lambda)
        var product = 1.0
        var count = 0
        while true {
            product *= source.nextUniform()
            if product <= threshold { return count }
            count += 1
        }
    }

    /// Hormann, "The transformed rejection method for generating Poisson random variables" (1993),
    /// in the PTRS form NumPy's `random_poisson_ptrs` uses.
    ///
    /// Constant expected cost in lambda, two uniforms per attempt, one `lgamma` when the squeeze
    /// misses. The discarded attempts are what make the sampler exact.
    @usableFromInline
    static func poissonTransformedRejection<R: RandomSource>(
        lambda: Double, _ source: inout R
    )
        -> Int
    {
        let logLambda = Foundation.log(lambda)
        let b = 0.931 + 2.53 * lambda.squareRoot()
        let a = -0.059 + 0.02483 * b
        let inverseAlpha = 1.1239 + 1.1328 / (b - 3.4)
        let squeezeBound = 0.9277 - 3.6224 / (b - 2.0)

        while true {
            let u = source.nextUniform() - 0.5
            let v = source.nextUniform()
            let us = 0.5 - abs(u)
            let candidate = ((2.0 * a / us + b) * u + lambda + 0.43).rounded(.down)

            if us >= 0.07 && v <= squeezeBound { return Int(candidate) }
            // `u` can be exactly -0.5, which makes `us` zero and `candidate` -infinity. That fails
            // the acceptance test above, and `candidate < 0` here rejects it before `Int(_:)` sees
            // it and traps.
            if candidate < 0 || (us < 0.013 && v > us) { continue }

            let lhs =
                Foundation.log(v) + Foundation.log(inverseAlpha)
                - Foundation.log(a / (us * us) + b)
            let rhs = -lambda + candidate * logLambda - lgamma(candidate + 1.0)
            if lhs <= rhs { return Int(candidate) }
        }
    }
}
