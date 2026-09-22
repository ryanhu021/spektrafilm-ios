"""Goldens for both interpolators, including the non-monotonic DIR-coupler axis and the
NaN policies."""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

@fixture
def interpolation():
    """np.interp on a monotonic and a genuinely non-monotonic axis, and fast_interp."""
    from opt_einsum import contract
    from spektrafilm.model.couplers import compute_dir_couplers_matrix
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.params_schema import DirCouplersParams
    from spektrafilm.utils.fast_interp import fast_interp

    matrix = compute_dir_couplers_matrix(DirCouplersParams())

    # kodak_portra_400 keeps the coupler axis ascending. fujifilm_velvia_100 is a positive stock,
    # where log_exposure - couplers_amount_curves steps backwards and np.interp's guess-threaded
    # search decides the answer.
    for stock in ["kodak_portra_400", "fujifilm_velvia_100"]:
        profile = load_profile(stock)
        curves = np.asarray(profile.data.density_curves)
        log_exposure = np.asarray(profile.data.log_exposure)
        normalised = curves - np.nanmin(curves, axis=0)
        positive = profile.info.type == "positive"
        silver = (np.nanmax(normalised, axis=0) - normalised) if positive else normalised.copy()
        axis = log_exposure[:, None] - contract("jk, km->jm", silver, matrix)
        # The Swift test rebuilds the interpolation from the axis and the curves, so ship both.
        yield f"interp_axis_{stock}", axis
        yield f"interp_curves_{stock}", curves
        result = np.zeros_like(curves)
        for c in range(3):
            if positive:
                result[:, c] = -np.interp(log_exposure, axis[:, c], -curves[:, c])
            else:
                result[:, c] = np.interp(log_exposure, axis[:, c], curves[:, c])
        yield f"interp_npinterp_{stock}", result

    # fast_interp with a shared axis and with a per-channel axis (the gamma_factor case).
    profile = load_profile("kodak_portra_400")
    log_exposure = np.asarray(profile.data.log_exposure)
    curves = np.asarray(profile.data.density_curves)
    yield "interp_log_exposure", log_exposure
    rng = np.random.default_rng(7)
    # Deliberately spans past both ends of the axis to pin the clamping.
    query = np.ascontiguousarray(rng.uniform(-5.0, 6.0, size=(16, 16, 3)))
    yield "interp_fast_query", query
    yield "interp_fast_shared_axis", fast_interp(query, log_exposure, curves)
    gamma = np.array([0.9, 1.0, 1.15])
    yield "interp_fast_perchannel_axis", fast_interp(
        query, log_exposure[:, None] / gamma[None, :], curves
    )

    # NaN policy. Neither behaviour is specified anywhere:
    #   fast_interp's NaN falls past both clamp tests into searchsorted, which returns K, so it
    #   indexes inv_dx one past the end. Numba's fastmath=True lets it assume NaN never occurs, and
    #   the left-clamp that comes out is an artefact of this toolchain.
    #   np.interp's single-point path is `x < xp ? left : (x > xp ? right : fp[0])`, and both
    #   comparisons are false for NaN, so NaN yields fp[0].
    # Cheap to pin, expensive to rediscover.
    nan_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    nan_values = np.tile(np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]).T, (1, 3))
    nan_query = np.full((1, 3, 3), np.nan)
    nan_query[0, 1, :] = 2.5
    nan_query[0, 2, :] = -99.0
    yield "interp_fast_nan_axis", nan_axis
    yield "interp_fast_nan_values", nan_values
    yield "interp_fast_nan_input", nan_query
    yield "interp_fast_nan_query", fast_interp(
        np.ascontiguousarray(nan_query), nan_axis, nan_values
    )
    yield "interp_npinterp_single_point", np.interp(
        np.array([np.nan, -1.0, 1.0, 5.0]), np.array([1.0]), np.array([7.0])
    )
