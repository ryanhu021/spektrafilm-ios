"""Goldens for the shared scalar primitives, the three boundary index maps and the spectral grid.

Nothing here is stochastic and almost everything is closed form, so the Swift side gates these at
1e-12 or tighter.

Non-finite values are deliberately kept out of the compared arrays. `Golden.swift`'s `parity()`
computes `abs(a - e)`, and `inf - inf` is NaN, which poisons the RMS instead of failing loudly; the
infinity cases are asserted directly in `NumericsTests.swift` against values measured here.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.ndimage as ndi

from fixture_registry import fixture

# (a, p) pairs for spow. Every result is finite and non-NaN, so the whole grid can be one golden.
# The negative bases are the point: np.power(-0.5, 2.2) is NaN, spow is not.
SPOW_CASES = [
    (0.5, 2.2),
    (-0.5, 2.2),
    (0.0, 2.2),
    (-0.0, 2.2),
    (0.0, 0.0),
    (2.0, 0.5),
    (-2.0, 0.5),
    (-8.0, 1.0 / 3.0),
    (-2.0, 0.15),
    (1.0, 0.0),
    (-1.0, 3.0),
    (16.0, 0.25),
    (-16.0, 0.25),
    (1e-300, 3.0),
    (-1e-300, 3.0),
    (0.18, 1.0 / 2.4),
    (-0.18, 1.0 / 2.4),
    (1.5, -2.0),
    (-1.5, -2.0),
    (255.0, 1.0),
]

# (a, b) pairs for np.fmax. NaN is included: it round-trips through the .spkg payload and the Swift
# comparison treats NaN as equal to NaN, so a Swift `max` that propagated NaN would show up as a
# NaN mismatch.
FMAX_CASES = [
    (1.0, 2.0),
    (2.0, 1.0),
    (-5.0, -7.0),
    (np.nan, 3.0),
    (3.0, np.nan),
    (np.nan, np.nan),
    (-0.0, 0.0),
    (0.0, -0.0),
    (-1e-12, 0.0),
    (1e-12, 0.0),
    (0.0, 0.0),
]

NAN_TO_NUM_CASES = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.5, -2.5, 1e308, -1e308]

# log10_guard inputs: the negatives and the NaN all land on -10, which is the floor the four call
# sites rely on.
LOG10_GUARD_CASES = [np.nan, -1.0, -1e-30, 0.0, 1e-30, 1e-10, 1e-5, 0.184, 1.0, 10.0, 1e6]

# Wide enough to wrap several periods on both sides, so a map that is only correct one step over
# fails. n = 1 pins the degenerate case: numpy.pad(mode='reflect') makes it a constant.
BOUNDARY_COUNTS = [1, 5, 8]


def _reflect_edge_duplicated(i: int, n: int) -> int:
    """Upstream `_reflect` from fast_gaussian_filter.py, for self-validation."""
    if 0 <= i < n:
        return i
    if -n <= i < 0:
        return -i - 1
    if n <= i < 2 * n:
        return 2 * n - 1 - i
    period = 2 * n
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - 1 - i
    return i


def _mirror_edge_shared(i: int, n: int) -> int:
    if n == 1:
        return 0
    period = 2 * (n - 1)
    k = i % period
    if k < 0:
        k += period
    if k >= n:
        k = period - k
    return k


def _clamp_edge(i: int, n: int) -> int:
    return min(max(i, 0), n - 1)


@fixture
def numerics():
    """spow, fmax, nan_to_num, the guarded log10, linspace and the NaN-skipping reductions."""
    from spektrafilm.profiles.io import load_profile
    from colour.algebra import spow

    spow_in = np.array(SPOW_CASES, dtype=np.float64)
    spow_out = np.array([spow(a, p) for a, p in SPOW_CASES], dtype=np.float64)
    assert np.isfinite(spow_out).all(), "a spow case produced a non-finite result"
    yield "num_spow_input", spow_in
    yield "num_spow", spow_out

    fmax_in = np.array(FMAX_CASES, dtype=np.float64)
    yield "num_fmax_input", fmax_in
    yield "num_fmax", np.fmax(fmax_in[:, 0], fmax_in[:, 1])

    nan_in = np.array(NAN_TO_NUM_CASES, dtype=np.float64)
    nan_out = np.nan_to_num(nan_in)
    assert np.isfinite(nan_out).all()
    yield "num_nan_to_num_input", nan_in
    yield "num_nan_to_num", nan_out

    guard_in = np.array(LOG10_GUARD_CASES, dtype=np.float64)
    yield "num_log10_guard_input", guard_in
    yield "num_log10_guard", np.log10(np.fmax(guard_in, 0.0) + 1e-10)

    # LOG_EXPOSURE itself, plus the shapes of linspace that the rest of the engine asks for.
    yield "num_linspace_log_exposure", np.linspace(-3, 4, 256)
    yield "num_linspace_cases", np.array(
        [
            np.linspace(0.0, 1.0, 17),
            np.linspace(1.0, 0.0, 17),
            np.linspace(-3.0, 4.0, 17),
            np.linspace(2.0, 2.0, 17),
            np.linspace(-np.pi, np.pi, 17),
            np.linspace(0.002, 0.18, 17),
        ]
    )
    # endpoint=False: the gamut-compression hue grid is built this way.
    yield "num_linspace_open_cases", np.array(
        [
            np.linspace(-np.pi, np.pi, 16, endpoint=False),
            np.linspace(0.0, 1.0, 16, endpoint=False),
        ]
    )
    yield "num_linspace_single", np.linspace(5.0, 9.0, 1)

    # Synthetic reduction input with an all-NaN column (2) and an all-NaN row (3), which is where
    # numpy warns and returns NaN.
    reduce_in = np.arange(36, dtype=np.float64).reshape(12, 3) / 7.0 - 1.0
    reduce_in[:, 2] = np.nan
    reduce_in[3, :] = np.nan
    reduce_in[0, 1] = np.nan
    reduce_in[11, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield "num_nan_reduce_input", reduce_in
        yield "num_nan_min_axis0", np.nanmin(reduce_in, axis=0)
        yield "num_nan_max_axis0", np.nanmax(reduce_in, axis=0)
        yield "num_nan_mean_axis0", np.nanmean(reduce_in, axis=0)
        yield "num_nan_mean_axis1", np.nanmean(reduce_in, axis=1)
        yield "num_nan_mean_flat", np.array([np.nanmean(reduce_in)])

        # The same reductions on real data. fujifilm_c200 is the one bundled profile whose
        # channel_density carries missing samples: 8/7/7 NaN per channel and 7 all-NaN rows.
        density = np.asarray(load_profile("fujifilm_c200").data.channel_density, dtype=np.float64)
        assert np.isnan(density).any(), "fujifilm_c200 channel_density lost its NaN"
        yield "num_nan_min_c200_channel_density", np.nanmin(density, axis=0)
        yield "num_nan_max_c200_channel_density", np.nanmax(density, axis=0)
        yield "num_nan_mean_c200_channel_density", np.nanmean(density, axis=0)
        yield "num_nan_mean_c200_per_wavelength", np.nanmean(density, axis=1)


@fixture
def boundary_index():
    """The three index maps, as padded ramps so the values are the indices."""
    for n in BOUNDARY_COUNTS:
        ramp = np.arange(n, dtype=np.float64)
        pad = 3 * n + 2
        for mode, name, fn in [
            ("symmetric", "reflect_edge_duplicated", _reflect_edge_duplicated),
            ("reflect", "mirror_edge_shared", _mirror_edge_shared),
            ("edge", "clamp_edge", _clamp_edge),
        ]:
            padded = np.pad(ramp, pad, mode=mode)
            expected = np.array([fn(i - pad, n) for i in range(n + 2 * pad)], dtype=np.float64)
            assert np.array_equal(padded, expected), f"{name} disagrees with numpy.pad {mode} at n={n}"
            yield f"boundary_{name}_n{n}", padded

    # scipy.ndimage's mode names for the same three maps, which are not numpy.pad's names.
    # correlate1d with a single-tap kernel reads the map off directly:
    # out[t, i] == ramp[map(i + t - pad)].
    n = 5
    pad = 3 * n + 2
    ramp = np.arange(n, dtype=np.float64)
    for mode, name, fn in [
        ("reflect", "reflect_edge_duplicated", _reflect_edge_duplicated),
        ("mirror", "mirror_edge_shared", _mirror_edge_shared),
        ("nearest", "clamp_edge", _clamp_edge),
    ]:
        rows = []
        for t in range(2 * pad + 1):
            weights = np.zeros(2 * pad + 1)
            weights[t] = 1.0
            rows.append(ndi.correlate1d(ramp, weights, mode=mode))
        table = np.array(rows)
        expected = np.array(
            [[fn(i + t - pad, n) for i in range(n)] for t in range(2 * pad + 1)], dtype=np.float64
        )
        assert np.array_equal(table, expected), f"scipy.ndimage {mode} is not {name}"
        yield f"boundary_ndimage_{name}_n{n}", table


@fixture
def spectral_shape():
    """The 81-point grid and the two spectral contractions, checked on the embedded CMFs."""
    from spektrafilm.config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS

    wavelengths = SPECTRAL_SHAPE.wavelengths
    assert wavelengths.shape == (81,)
    yield "spectralshape_wavelengths_nm", wavelengths
    yield "spectralshape_wavelengths_m", wavelengths * 1e-9

    cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:], dtype=np.float64)
    assert cmfs.shape == (81, 3)
    # A spectrum with no repeated values, so a transposed or reversed contraction cannot coincide.
    probe = wavelengths * 1e-3
    yield "spectralshape_contract_cmfs", np.einsum("k,kl->l", probe, cmfs)
    yield "spectralshape_cmfs_column_sums", cmfs.sum(axis=0)
    yield "spectralshape_cmfs_weighted_rows", np.einsum("l,kl->k", np.array([0.2, 0.5, 0.3]), cmfs)
    yield "spectralshape_cmfs_times_probe", (cmfs * probe[:, None]).ravel()
