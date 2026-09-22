#!/usr/bin/env python3
"""Generate the parity goldens the Swift tests check against.

Every fixture comes from the pinned oracle (see upstream_pin.json) and is committed, so CI needs no
Python. Only re-run this after `make oracle`. A changed fixture means the render changed, so review
the delta before committing it.

Add a fixture with a `@fixture`-decorated function returning `(name, array)` pairs. Keep inputs
small and deterministic, since these files live in git.

Usage:  Tools/parity/oracle/.venv/bin/python Tools/parity/generate_goldens.py [name ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_env  # noqa: E402
import spkg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GOLDENS = REPO / "Tests" / "SpektraFilmTests" / "Goldens"

_FIXTURES: list = []


def fixture(fn):
    _FIXTURES.append(fn)
    return fn


# Covers the awkward parts of every transfer function: below zero, where signed powers and
# NaN-producing gamma functions disagree; the piecewise breakpoints; and above 1.0, where the
# simulation routinely lands before gamut compression.
TRANSFER_SWEEP = np.concatenate([
    np.array([-1.0, -0.5, -0.04045, -0.0031308, -1e-9, 0.0]),
    np.array([1e-9, 1e-6, 0.0031308, 0.00390625, 0.001953125, 0.018, 0.04045, 0.081]),
    np.linspace(0.0, 1.0, 65),
    np.array([1.0, 1.0000001, 1.5, 2.0, 8.0, 100.0]),
])


@fixture
def transfer_functions():
    """cctf encode/decode per colourspace, over TRANSFER_SWEEP."""
    import colour

    yield "transfer_sweep_input", TRANSFER_SWEEP
    for name in [
        "sRGB", "DCI-P3", "Display P3", "Adobe RGB (1998)",
        "ITU-R BT.2020", "ProPhoto RGB", "ACES2065-1",
    ]:
        cs = colour.RGB_COLOURSPACES[name]
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        slug = slug.replace("-", "_")
        with colour.utilities.domain_range_scale("ignore"):
            # Negative bases with fractional exponents produce NaN here by design, under colour's
            # "Indeterminate" handling, so the warning is noise.
            with np.errstate(invalid="ignore"):
                yield f"transfer_{slug}_encode", np.asarray(cs.cctf_encoding(TRANSFER_SWEEP))
                yield f"transfer_{slug}_decode", np.asarray(cs.cctf_decoding(TRANSFER_SWEEP))


@fixture
def illuminants():
    """Normalised illuminant spectra and their chromaticities."""
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.utils.spectral_upsampling import _illuminant_to_xy

    labels = ["D50", "D55", "D65", "T", "K75P", "TH-KG3", "TH-KG3-L", "BB3400", "BB5500"]
    yield "illuminant_spectra", np.stack([standard_illuminant(x) for x in labels])
    yield "illuminant_xy", np.stack([_illuminant_to_xy(x) for x in labels])


@fixture
def observer():
    """The aligned colour-matching functions and cone fundamentals."""
    from spektrafilm.config import STANDARD_OBSERVER_CMFS, STANDARD_OBSERVER_LMS

    yield "observer_cmfs", np.asarray(STANDARD_OBSERVER_CMFS[:])
    yield "observer_lms", np.asarray(STANDARD_OBSERVER_LMS[:])


@fixture
def filters():
    """Resampled filter transmittances and the analytic band-pass filter."""
    from spektrafilm.model.color_filters import (
        compute_band_pass_filter,
        custom_dichroic_filters,
        durst_digital_light_dicrhoic_filters,
        edmund_optics_dichroic_filters,
        generic_lens_transmission,
        schott_kg3_heat_filter,
        thorlabs_dichroic_filters,
    )

    yield "filter_schott_kg3", np.asarray(schott_kg3_heat_filter.transmittance)
    yield "filter_canon_lens", np.asarray(generic_lens_transmission.transmittance)
    yield "filter_dichroic_custom", np.asarray(custom_dichroic_filters.filters)
    yield "filter_dichroic_thorlabs", np.asarray(thorlabs_dichroic_filters.filters)
    yield "filter_dichroic_edmund", np.asarray(edmund_optics_dichroic_filters.filters)
    yield "filter_dichroic_durst", np.asarray(durst_digital_light_dicrhoic_filters.filters)
    # Default band-pass is inert (amplitude 0); these are the engaged settings.
    yield "filter_bandpass_default", compute_band_pass_filter((0.0, 410.0, 8.0), (0.0, 675.0, 15.0))
    yield "filter_bandpass_active", compute_band_pass_filter((1.0, 410.0, 8.0), (1.0, 675.0, 15.0))
    yield "filter_bandpass_partial", compute_band_pass_filter((0.6, 395.0, 12.0), (0.35, 690.0, 9.0))


@fixture
def colour_conversions():
    """XYZ→RGB with chromatic adaptation, and the same-space RGB_to_RGB path."""
    import colour
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.config import STANDARD_OBSERVER_CMFS
    from opt_einsum import contract

    rng = np.random.default_rng(20260922)
    xyz = rng.uniform(-0.2, 1.4, size=(8, 8, 3))
    yield "colour_xyz_input", xyz

    for label in ["D50", "K75P"]:
        illuminant = standard_illuminant(label)
        norm = np.sum(illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
        illuminant_xyz = contract("k,kl->l", illuminant, STANDARD_OBSERVER_CMFS[:]) / norm
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        for cs in ["sRGB", "Display P3", "ITU-R BT.2020", "ProPhoto RGB"]:
            slug = cs.lower().replace(" ", "_").replace("-", "_").replace(".", "")
            rgb = colour.XYZ_to_RGB(
                xyz, colourspace=cs, apply_cctf_encoding=False, illuminant=illuminant_xy
            )
            yield f"colour_xyz_to_rgb_{slug}_{label.lower()}", np.asarray(rgb)

    # Upstream runs the output transfer function through RGB_to_RGB with input == output, which
    # still multiplies by fromXYZ·(CAT·toXYZ), which is near identity but not identity.
    rgb = rng.uniform(0.0, 1.0, size=(8, 8, 3))
    yield "colour_rgb_input", rgb
    for cs in ["sRGB", "Display P3", "ProPhoto RGB"]:
        slug = cs.lower().replace(" ", "_").replace("-", "_")
        with np.errstate(invalid="ignore"):
            out = colour.RGB_to_RGB(
                rgb, cs, cs, apply_cctf_decoding=False, apply_cctf_encoding=True
            )
        yield f"colour_same_space_encode_{slug}", np.asarray(out)


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


def main() -> int:
    oracle_env.require()
    wanted = set(sys.argv[1:])
    manifest: dict[str, dict] = {}
    total = 0
    for fn in _FIXTURES:
        if wanted and fn.__name__ not in wanted:
            continue
        for name, array in fn():
            array = np.asarray(array, dtype=np.float64)
            size = spkg.write(GOLDENS / f"{name}.spkg", array)
            manifest[name] = {
                "shape": list(array.shape),
                "bytes": size,
                "group": fn.__name__,
                "nan_count": int(np.isnan(array).sum()),
            }
            total += size
            print(f"  {name:48s} {str(array.shape):18s} {size:>9,d} B")

    manifest_path = GOLDENS / "manifest.json"
    if wanted and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.update(manifest)
        manifest = existing
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    print(f"\n{len(manifest)} fixtures, {total:,} B written this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
