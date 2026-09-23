"""Goldens for the error function, the CDFs built on it, and Brent root finding.

The erf sweep reaches into both tails because the Hanatos window divides by a sensitivity-weighted
integral of itself, so an erf that is only accurate near zero would survive a narrow sweep.

The root-finder cases use `math.*` (libm on Darwin) and explicit multiplications, never `**` or
`scipy.special`, so the Swift side evaluates bit-identical residuals and the roots can be compared
far below the 1e-4 gate.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
from scipy.optimize import brentq

from fixture_registry import fixture

# scipy.optimize.brentq defaults. The morph curves pass xtol=1e-10 and leave the rest alone.
_RTOL = 4.0 * np.finfo(float).eps
_MAXITER = 100

_BRENTQ_CASES = [
    (lambda x: x * x - 2.0, 0.0, 2.0),
    (lambda x: math.cos(x) - x, 0.0, 1.0),
    (lambda x: x * x * x - 2.0 * x - 5.0, 2.0, 3.0),
    (lambda x: math.exp(x) - 3.0 * x, 0.0, 1.0),
    (lambda x: math.atan(x) - 0.5, -1.0, 2.0),
    (lambda x: x * x * x - 0.027, -1.0, 0.5),
    (lambda x: math.tanh(x - 1.0), -5.0, 5.0),
    (lambda x: x * x * x * x * x * x * x * x * x - 1e-3, -1.0, 1.3),
    (lambda x: 1.0 / (x - 0.3) - 2.0, 0.35, 5.0),
    (lambda x: math.exp(-x * x) - 0.75, 0.0, 3.0),
]

# Roots outside the reference's initial [-0.25, 0.25], so the doubling search has to run. The last
# three pin conventions that a plausible rewrite gets wrong by 1e-13 or misses entirely: the r_lo
# and r_hi exact-zero short-circuits, and the twelfth bracket. Dropping either short-circuit returns
# 0.24999999999989636; stopping at eleven doublings gives up on the +-512 case.
_BRACKET_CASES = [
    lambda x: math.tanh(x - 1.7),
    lambda x: 1.0 - math.exp(-(x + 1.1)),
    lambda x: math.tanh(0.3 * (x - 3.8)),
    lambda x: (x + 0.25) * (x - 3.0),
    lambda x: (x - 0.25) * (x + 3.0),
    lambda x: math.tanh(0.001 * (x - 300.0)),
]


def _expanding_bracket_root(f, xtol=1e-10):
    """`morph_curves._developer_exhaustion_center_offset`'s bracket search, verbatim."""
    lo, hi = -0.25, 0.25
    r_lo, r_hi = f(lo), f(hi)
    for _ in range(12):
        if r_lo == 0.0:
            return lo
        if r_hi == 0.0:
            return hi
        if r_lo * r_hi < 0.0:
            return float(brentq(f, lo, hi, xtol=xtol, rtol=_RTOL, maxiter=_MAXITER))
        lo *= 2.0
        hi *= 2.0
        r_lo, r_hi = f(lo), f(hi)
    return None


def _sweep():
    return np.unique(
        np.concatenate(
            [
                np.linspace(-6.0, 6.0, 601),
                np.linspace(-45.0, 45.0, 181),
                np.logspace(-20.0, 1.5, 200),
                -np.logspace(-20.0, 1.5, 200),
                # scipy's erfc first underflows to 0 at 26.6417475570463; Darwin still returns
                # subnormals there and out to 27, so 27.0 pins that divergence.
                np.array([0.0, 0.5, 1.0, 26.5, 26.55, 27.0, -26.5, 1.0 / math.sqrt(2.0)]),
            ]
        )
    )


@fixture
def transcendental():
    """erf, erfc, the normal and Gumbel CDFs, Brent roots, and the morph exhaustion offsets."""
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.utils.morph_curves import _developer_exhaustion_center_offset
    from spektrafilm.utils.morph_curves import _gumbel_matched_cdf

    xs = _sweep()
    yield "transcendental_sweep_input", xs
    yield "transcendental_erf", scipy.special.erf(xs)
    yield "transcendental_erfc", scipy.special.erfc(xs)
    # scipy.stats.norm.cdf is scipy.special.ndtr, which is what _layer_cdf calls.
    yield "transcendental_normal_cdf", scipy.special.ndtr(xs)
    yield "transcendental_gumbel_cdf", _gumbel_matched_cdf(xs)

    roots = np.empty((2, len(_BRENTQ_CASES)))
    for row, xtol in enumerate([2e-12, 1e-10]):
        for col, (f, a, b) in enumerate(_BRENTQ_CASES):
            roots[row, col] = brentq(f, a, b, xtol=xtol, rtol=_RTOL, maxiter=_MAXITER)
    yield "transcendental_brentq_roots", roots

    yield "transcendental_bracket_roots", np.array(
        [_expanding_bracket_root(f) for f in _BRACKET_CASES]
    )

    # The whole composition the morph curves need: normal CDF, Gumbel CDF, bracket search, Brent.
    # kodak_portra_endura reports info.type == "negative"; both signed branches are pinned because
    # apply_print_curves_morph's signature defaults to "positive".
    model = load_profile("kodak_portra_endura").data.density_curves_model
    centers = np.asarray(model.centers, dtype=float)
    amplitudes = np.asarray(model.amplitudes, dtype=float)
    sigmas = np.asarray(model.sigmas, dtype=float)
    yield "transcendental_morph_centers", centers
    yield "transcendental_morph_amplitudes", amplitudes
    yield "transcendental_morph_sigmas", sigmas

    exhaustions = [0.1, 0.35, 1.0]
    offsets = np.empty((2, len(exhaustions), centers.shape[0]))
    for i, profile_type in enumerate(["positive", "negative"]):
        for j, exhaustion in enumerate(exhaustions):
            for c in range(centers.shape[0]):
                offsets[i, j, c] = _developer_exhaustion_center_offset(
                    centers[c],
                    amplitudes[c],
                    sigmas[c],
                    profile_type,
                    np.full(centers.shape[1], exhaustion),
                )
    yield "transcendental_morph_offsets", offsets
