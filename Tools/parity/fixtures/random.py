"""Goldens for the counter-based RNG and the distributions grain and glare draw from.

Byte-exact parity is impossible here: the Swift engine uses Philox4x32-10 and the reference uses
NumPy's MT19937 (and, on the `use_fast_stats` path, a Numba thread-local stream that no seed
reaches). So these fixtures carry two different kinds of number.

Deterministic, gated tight:
  the log-space parameter inversion of `fast_lognormal_from_mean_std`, which is pure arithmetic.

Statistical, used to calibrate tolerances:
  one realisation of the oracle's sampler, plus the standard deviation of each statistic across
  `REPEATS` independent realisations. The Swift test gates its own single realisation against the
  *theoretical* moment at 5x the measured standard deviation, so the oracle sets the noise floor
  without its RNG stream entering the comparison.

`fast_lognormal_from_mean_std` itself cannot be sampled into a committed fixture: its Numba kernel
has no seeding hook, so the numbers would change on every regeneration. `Generator.lognormal` with
mu and sigma from the reference's own inversion is the same distribution and is reproducible, so
that is what the lognormal moments come from.
"""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

SAMPLES = 512 * 512
REPEATS = 256
SEED = 20260922

# Spans both branches of the Swift sampler (Knuth below 10, transformed rejection at and above)
# and the range the grain model actually reaches: lambda ~ 4.5 at the low end, up to ~1.2e8 when
# `uniformity` is 1 and the density saturates (grain.md section 4.1).
POISSON_LAMBDAS = np.array(
    [0.5, 2.0, 4.5, 9.0, 9.999, 10.0, 12.0, 20.0, 30.0, 1.0e3, 1.0e6, 1.2e8]
)

# Small enough that the count distribution is checkable bin by bin, so these get a chi-square
# goodness of fit. 12.0 and 20.0 are there to put the transformed-rejection branch under the same
# test as Knuth; moments alone would not notice a wrong rejection bound.
PMF_LAMBDAS = np.array([0.5, 2.0, 4.5, 9.0, 12.0, 20.0])
PMF_BINS = 72

# (linear mean, linear std) pairs for the lognormal inversion. Covers the two production call
# sites, grain's clumping field at its default and forced-on settings and glare's (0.03, 0.7*0.03),
# plus the two degenerate branches: `mean <= 0` yields mu = sigma = 0, and `sigma < 1e-6` skips the
# normal draw entirely.
LOGNORMAL_PARAMS = np.array(
    [
        [1.0, 0.3],
        [1.0, 1.0],
        [1.0, 0.003428571428571428],
        [0.03, 0.021],
        [2.0, 0.5],
        [5.0, 2.0],
        [1.0, 0.0],
        [1.0, 1.0e-9],
        [0.0, 0.5],
        [-1.0, 0.5],
    ]
)

# The rows above that actually draw a normal variate, so their sample moments are non-degenerate.
LOGNORMAL_SAMPLED = np.array(
    [
        [1.0, 0.3],
        [1.0, 1.0],
        [1.0, 0.003428571428571428],
        [0.03, 0.021],
        [2.0, 0.5],
        [5.0, 2.0],
    ]
)


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, variance, skewness and excess kurtosis, all biased (divide by n) estimators."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean()
    centred = x - mean
    m2 = (centred**2).mean()
    m3 = (centred**3).mean()
    m4 = (centred**4).mean()
    skew = m3 / m2**1.5 if m2 > 0 else 0.0
    kurt = m4 / m2**2 - 3.0 if m2 > 0 else 0.0
    return mean, m2, skew, kurt


@fixture
def random_poisson():
    """Poisson moments from `Generator.poisson`, one realisation plus the sampling noise floor."""
    yield "random_poisson_lambdas", POISSON_LAMBDAS

    single = np.zeros((POISSON_LAMBDAS.size, 4))
    spread = np.zeros((POISSON_LAMBDAS.size, 4))
    for i, lam in enumerate(POISSON_LAMBDAS):
        rng = np.random.default_rng([SEED, i])
        single[i] = _moments(rng.poisson(lam, SAMPLES))
        repeats = np.array([_moments(rng.poisson(lam, SAMPLES)) for _ in range(REPEATS)])
        spread[i] = repeats.std(axis=0, ddof=1)
    yield "random_poisson_moments", single
    yield "random_poisson_moment_sd", spread


@fixture
def random_poisson_goodness_of_fit():
    """Exact Poisson mass functions and the chi-square critical values to test them against."""
    from scipy.stats import chi2, kstwobign, poisson

    yield "random_poisson_pmf_lambdas", PMF_LAMBDAS

    # Column PMF_BINS holds the remaining tail mass, so every row sums to 1 and the Swift side can
    # pool bins by expected count without needing a second fixture.
    pmf = np.zeros((PMF_LAMBDAS.size, PMF_BINS + 1))
    counts = np.arange(PMF_BINS)
    for i, lam in enumerate(PMF_LAMBDAS):
        pmf[i, :PMF_BINS] = poisson.pmf(counts, lam)
        pmf[i, PMF_BINS] = poisson.sf(PMF_BINS - 1, lam)
    yield "random_poisson_pmf", pmf

    # Indexed by degrees of freedom minus one. p = 1e-3, so a correct sampler fails about once in
    # a thousand suite runs per assertion, and a wrong one fails every time.
    yield "random_chi2_critical_1em3", chi2.ppf(1.0 - 1e-3, np.arange(1, 129))

    # Asymptotic two-sided Kolmogorov-Smirnov critical value; divide by sqrt(n) for the D gate.
    yield "random_ks_critical_1em3", np.array([kstwobign.ppf(1.0 - 1e-3)])


@fixture
def random_normal():
    """Standard normal moments and their sampling noise floor, for the Box-Muller draw."""
    rng = np.random.default_rng([SEED, 1000])
    single = np.array(_moments(rng.standard_normal(SAMPLES)))
    repeats = np.array([_moments(rng.standard_normal(SAMPLES)) for _ in range(REPEATS)])
    yield "random_normal_moments", single
    yield "random_normal_moment_sd", repeats.std(axis=0, ddof=1)


@fixture
def random_lognormal():
    """The deterministic log-space inversion, and the moments it implies."""
    from spektrafilm.utils.fast_stats import fast_lognormal_from_mean_std

    # Transcribed from fast_stats.py:166 so the golden is the reference's arithmetic, not a
    # restatement of the textbook identity.
    def log_params(mean: float, std: float) -> tuple[float, float]:
        if mean <= 0:
            return 0.0, 0.0
        sigma2 = np.log(1.0 + (std * std) / (mean * mean))
        return np.log(mean) - sigma2 / 2.0, np.sqrt(sigma2)

    table = np.zeros((LOGNORMAL_PARAMS.shape[0], 4))
    for i, (mean, std) in enumerate(LOGNORMAL_PARAMS):
        mu, sigma = log_params(mean, std)
        table[i] = [mean, std, mu, sigma]
    yield "random_lognormal_log_params", table

    single = np.zeros((LOGNORMAL_SAMPLED.shape[0], 4))
    spread = np.zeros((LOGNORMAL_SAMPLED.shape[0], 4))
    for i, (mean, std) in enumerate(LOGNORMAL_SAMPLED):
        mu, sigma = log_params(mean, std)
        rng = np.random.default_rng([SEED, 2000, i])
        single[i] = _moments(rng.lognormal(mu, sigma, SAMPLES))
        repeats = np.array(
            [_moments(rng.lognormal(mu, sigma, SAMPLES)) for _ in range(REPEATS)]
        )
        spread[i] = repeats.std(axis=0, ddof=1)
    yield "random_lognormal_params_sampled", LOGNORMAL_SAMPLED
    yield "random_lognormal_moments", single
    yield "random_lognormal_moment_sd", spread

    # The reference kernel is non-reproducible, so it cannot be a committed fixture. Comparing its
    # moments against the Generator.lognormal row above at generation time still catches a
    # misreading of the inversion, and the tolerance is loose enough to be stable.
    for i, (mean, std) in enumerate(LOGNORMAL_SAMPLED):
        sampled = fast_lognormal_from_mean_std(
            np.full(SAMPLES, mean), np.full(SAMPLES, std)
        )
        got_mean, got_var, _, _ = _moments(sampled)
        assert abs(got_mean - single[i, 0]) < 20.0 * spread[i, 0], (mean, std, got_mean)
        assert abs(got_var - single[i, 1]) < 20.0 * spread[i, 1], (mean, std, got_var)
