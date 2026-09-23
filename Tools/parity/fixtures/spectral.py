"""Goldens for Hanatos 2025 spectral upsampling: RGB to per-film camera raw.

Covers the chromaticity warp, the shipped 192x192x81 irradiance LUT, the sensitivity adaptation
window and surface, the per-film tc_lut contraction, the Mitchell 2D LUT fetch and the end-to-end
raw. Input gamut compression is out of scope here, so every tc_lut is built with it bypassed.

Two fixtures hold the full 192x192x3 array because they are the main gates: the production tc_lut
and the on-grid Mitchell fetch. The variants are decimated on a fixed index list that includes both
ends of each axis, so a transposed or off-by-one LUT still fails.
"""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

FILM = "kodak_portra_400"
FILM_T = "kodak_vision3_500t"

# Every 8th grid index plus the last one, so index 0 and index 191 are both sampled on both axes.
DECIMATION = list(range(0, 192, 8)) + [191]

# The six cells the spec pins, in the order the Swift test reads them.
CELLS = [(0, 0), (0, 191), (191, 0), (191, 191), (96, 96), (85, 99)]

# The production patch from the spec, plus eight patches that leave the visible locus: negative
# brightness, super-unit chromaticity, zero, and a NaN channel.
RGB_PRODUCTION = np.array(
    [
        [[0.00, 0.00, 0.000], [0.184, 0.184, 0.184], [1.0, 1.0, 1.00], [0.80, 0.20, 0.100]],
        [[0.10, 0.70, 0.300], [0.050, 0.050, 0.500], [1.5, 0.3, 0.05], [0.02, 0.01, 0.005]],
    ]
)
RGB_EXTREME = np.array(
    [
        [[-0.5, 1.0, -0.2], [1.5, -0.2, -0.1], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [[-1.0, -1.0, -1.0], [1e-12, 1e-12, 1e-12], [np.nan, 0.5, 0.5], [1.0, 0.0, 0.0]],
    ]
)
RGB_PATCH = np.concatenate((RGB_PRODUCTION, RGB_EXTREME), axis=0)

COLOUR_SPACES = [
    "sRGB",
    "DCI-P3",
    "Display P3",
    "Adobe RGB (1998)",
    "ITU-R BT.2020",
    "ProPhoto RGB",
    "ACES2065-1",
]
ILLUMINANT_LABELS = ["D65", "D55", "D50", "T", "TH-KG3", "TH-KG3-L", "BB3400", "K75P"]


def _decimate(array):
    """Samples a 192x192x... array on DECIMATION along both grid axes."""
    index = np.ix_(DECIMATION, DECIMATION)
    return array[index]


def _sensitivity(stock):
    from spektrafilm.profiles.io import load_profile

    profile = load_profile(stock)
    return np.nan_to_num(10.0 ** np.asarray(profile.data.log_sensitivity)), profile


def _adaptation(profile, *, window=True, surface=False, blur=0.0):
    from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation

    return Hanatos2025SensitivityAdaptation(
        window_params=np.asarray(profile.data.hanatos2025_adaptation_window_params),
        surface_params=np.asarray(profile.data.hanatos2025_adaptation_surface_params),
        spectral_gaussian_blur=blur,
        reference_illuminant=profile.info.reference_illuminant,
        apply_window=window,
        apply_surface=surface,
    )


@fixture
def spectral_irradiance_lut():
    """The shipped 192x192x81 float16 table, read through the same path the engine uses."""
    from spektrafilm.utils.spectral_upsampling import HANATOS2025_SPECTRA_LUT as LUT

    yield "su_spectra_lut_cells", np.stack([LUT[i, j] for i, j in CELLS])
    # The total is 3.2e6 over 3M elements, so a sequential Swift sum and NumPy's pairwise sum differ
    # by ~2e-3. Scaled by 1e-6, the sum still catches a misread element and stays within the gate.
    yield "su_spectra_lut_stats_scaled", np.array([LUT.min(), LUT.max(), LUT.sum() * 1e-6])


@fixture
def spectral_chromaticity():
    """_tri2quad / _quad2tri, including the x > 1 fold and the 1e-10 guard."""
    from spektrafilm.utils.spectral_upsampling import _quad2tri, _tri2quad

    xy = np.array(
        [
            [1 / 3, 1 / 3],
            [0.3127, 0.3290],
            [0.332431638921549, 0.347444398930646],
            [0.64, 0.33],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.2, -0.1],
            [0.999999, 0.5],
        ]
    )
    yield "su_tri2quad_input", xy
    yield "su_tri2quad", _tri2quad(xy)
    yield "su_quad2tri", _quad2tri(_tri2quad(xy))


@fixture
def spectral_illuminant_chromaticity():
    """_illuminant_to_xy and its tc for every illuminant a bundled profile references."""
    from spektrafilm.utils.spectral_upsampling import _illuminant_to_xy, _tri2quad

    xy = np.stack([_illuminant_to_xy(label) for label in ILLUMINANT_LABELS])
    yield "su_illuminant_xy", xy
    yield "su_illuminant_tc", _tri2quad(xy)


@fixture
def spectral_rgb_to_tc_b():
    """The composed CAT16 matrix per (colour space, reference illuminant), then tc and b."""
    import colour

    from spektrafilm.utils.spectral_upsampling import _illuminant_to_xy, _rgb_to_tc_b

    matrices = []
    for label in ["D55", "T"]:
        illu_xy = _illuminant_to_xy(label)
        for space in COLOUR_SPACES:
            matrices.append(
                colour.RGB_to_XYZ(
                    np.eye(3),
                    colourspace=space,
                    apply_cctf_decoding=False,
                    illuminant=illu_xy,
                    chromatic_adaptation_transform="CAT16",
                ).T
            )
    yield "su_rgb_to_tc_b_matrices", np.stack(matrices)

    yield "su_tcb_input", RGB_PATCH
    for space, tag in (("ProPhoto RGB", "prophoto"), ("ACES2065-1", "aces")):
        tc, b = _rgb_to_tc_b(RGB_PATCH, space, False, "D55")
        yield f"su_tcb_{tag}_tc", tc
        yield f"su_tcb_{tag}_b", b


@fixture
def spectral_adaptation_window():
    """erf4 (shipped), its white-preserving normalisation, and the unreachable logiflex8."""
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.utils.spectral_upsampling import (
        eval_erf4_spectral_bandpass,
        eval_logiflex8_spectral_bandpass,
    )

    sensitivity, profile = _sensitivity(FILM)
    params = np.asarray(profile.data.hanatos2025_adaptation_window_params)
    window = eval_erf4_spectral_bandpass(params)
    yield "su_erf4_window", window

    illuminant = standard_illuminant("D55")
    normalization = np.sum(sensitivity * illuminant[:, None] * window, axis=0) / np.sum(
        sensitivity * illuminant[:, None], axis=0
    )
    yield "su_erf4_normalization", normalization
    yield "su_erf4_window_normalised", window / normalization

    yield (
        "su_logiflex8_window",
        eval_logiflex8_spectral_bandpass(np.array([415.0, 12.0, 667.0, 76.0, 430.0, 650.0, 1.0, 1.0])),
    )


@fixture
def spectral_adaptation_surface():
    """poly4, and the dead poly4_warp_xy model with synthetic alphas."""
    from spektrafilm.utils.spectral_upsampling import (
        _illuminant_to_xy,
        eval_poly4_log_exposure_surface,
        eval_poly4_warp_log_exposure_surface,
    )

    _, profile = _sensitivity(FILM)
    params = np.asarray(profile.data.hanatos2025_adaptation_surface_params)
    illu_xy = _illuminant_to_xy("D55")

    surface = eval_poly4_log_exposure_surface(params, illu_xy)
    yield "su_poly4_surface", _decimate(surface)
    yield "su_poly4_surface_cells", np.stack([surface[i, j] for i, j in CELLS])
    yield "su_poly4_surface_extrema", np.stack([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))])

    warped = eval_poly4_warp_log_exposure_surface(
        np.hstack([params, np.array([[0.5], [0.5], [0.5]])]), illu_xy
    )
    yield "su_poly4_warp_surface", _decimate(warped)


@fixture
def spectral_blur():
    """scipy.ndimage.gaussian_filter along the wavelength axis only: radius, kernel, reflect fold.

    The sigma is in array samples, so on a 5 nm grid sigma = 4 is 20 nm of blur. Two comments in
    the reference call it nm; they are wrong.
    """
    import scipy.ndimage

    from spektrafilm.utils.spectral_upsampling import HANATOS2025_SPECTRA_LUT as LUT

    corner = np.array(LUT[:4, :4, :], dtype=np.float64)
    for sigma in (1.0, 2.0, 4.0):
        tag = str(sigma).replace(".", "p")
        yield f"su_spectral_blur_s{tag}", scipy.ndimage.gaussian_filter(corner, (0, 0, sigma))

    delta = np.zeros((1, 1, 81))
    delta[0, 0, 0] = 1.0
    yield (
        "su_spectral_blur_delta",
        np.stack(
            [
                scipy.ndimage.gaussian_filter(delta, (0, 0, sigma))[0, 0]
                for sigma in (1.0, 2.0, 4.0)
            ]
        ),
    )


@fixture
def spectral_tc_lut():
    """compute_hanatos2025_tc_lut with compression bypassed, across the reachable branches."""
    from spektrafilm.utils.spectral_upsampling import (
        HANATOS2025_NO_ADAPTATION,
        compute_hanatos2025_tc_lut,
    )

    sensitivity, profile = _sensitivity(FILM)

    window_only = compute_hanatos2025_tc_lut(sensitivity, _adaptation(profile))
    yield "su_tc_lut_window_only", window_only
    yield "su_tc_lut_window_only_cells", np.stack([window_only[i, j] for i, j in CELLS])

    yield (
        "su_tc_lut_window_surface",
        _decimate(compute_hanatos2025_tc_lut(sensitivity, _adaptation(profile, surface=True))),
    )
    yield (
        "su_tc_lut_blur4",
        _decimate(compute_hanatos2025_tc_lut(sensitivity, _adaptation(profile, blur=4.0))),
    )
    yield (
        "su_tc_lut_no_adaptation",
        _decimate(compute_hanatos2025_tc_lut(sensitivity, HANATOS2025_NO_ADAPTATION)),
    )

    sensitivity_t, profile_t = _sensitivity(FILM_T)
    assert profile_t.info.reference_illuminant == "T", profile_t.info.reference_illuminant
    yield (
        "su_tc_lut_vision3_500t",
        _decimate(compute_hanatos2025_tc_lut(sensitivity_t, _adaptation(profile_t))),
    )


@fixture
def spectral_lut2d_mitchell():
    """apply_lut_cubic_2d: the Mitchell B=C=1/3 kernel, the mirror fold, the L<2 fallback."""
    from spektrafilm.utils.fast_interp_lut import apply_lut_cubic_2d, mitchell_weight
    from spektrafilm.utils.spectral_upsampling import compute_hanatos2025_tc_lut

    sensitivity, profile = _sensitivity(FILM)
    tc_lut = compute_hanatos2025_tc_lut(sensitivity, _adaptation(profile))

    yield "su_mitchell_weights", np.array([mitchell_weight(t) for t in np.linspace(-2.25, 2.25, 37)])

    # Exact grid coordinates. Mitchell does not interpolate, so this is not the stored LUT.
    base = np.linspace(0, 1, 192)
    grid = np.stack(np.meshgrid(base, base, indexing="ij"), axis=-1)
    yield "su_lut2d_grid", apply_lut_cubic_2d(tc_lut, grid)

    random_tc = np.random.default_rng(0).random((64, 64, 2))
    yield "su_lut2d_random_input", random_tc
    yield "su_lut2d_random", apply_lut_cubic_2d(tc_lut, random_tc)

    edges = np.array(
        [
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
                [-0.001, 0.5],
                [1.001, 0.5],
                [0.5, -0.001],
                [0.5, 1.001],
                [1.0 - 1.0 / 191.0, 0.5],
                [0.5, 1.0 / 191.0],
            ]
        ]
    )
    yield "su_lut2d_edges_input", edges
    yield "su_lut2d_edges", apply_lut_cubic_2d(tc_lut, edges)

    degenerate_lut = np.ones((1, 1, 3)) * np.array([1.0, 2.0, 3.0])
    degenerate_tc = np.random.default_rng(7).random((4, 4, 2))
    yield "su_lut2d_degenerate_input", degenerate_tc
    yield "su_lut2d_degenerate", apply_lut_cubic_2d(degenerate_lut, degenerate_tc)


@fixture
def spectral_rgb_to_raw():
    """rgb_to_raw_hanatos2025 end to end, with and without a prebuilt tc_lut."""
    from spektrafilm.utils.spectral_upsampling import (
        compute_hanatos2025_tc_lut,
        rgb_to_raw_hanatos2025,
    )

    sensitivity, profile = _sensitivity(FILM)
    tc_lut = compute_hanatos2025_tc_lut(sensitivity, _adaptation(profile))

    yield (
        "su_raw_hanatos2025",
        rgb_to_raw_hanatos2025(RGB_PATCH, sensitivity, "ProPhoto RGB", False, "D55", tc_lut=tc_lut),
    )
    yield (
        "su_raw_hanatos2025_fallback",
        rgb_to_raw_hanatos2025(RGB_PATCH, sensitivity, "ProPhoto RGB", False, "D55"),
    )
