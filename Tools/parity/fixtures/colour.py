"""Goldens for the colour foundations: transfer functions, illuminants, observers, filters
and the chromatic-adaptation conversions."""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

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
