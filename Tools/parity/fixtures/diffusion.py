"""Goldens for diffusion, halation and blur.

Calls `model/diffusion.py`, `utils/fast_gaussian_filter.py` and `utils/numba_boost_hightlights.py`
directly, never through `simulate()`: the surrounding stages would add parity risk that has nothing
to do with this subsystem and would make a failure hard to localise.

Pixel sizes come from `film_format_mm * 1000 / max(H, W)` at 35 mm, so `PIXEL_SIZE_UM[edge]` is what
a frame with that long edge would use. The FIR/IIR dispatch and the PSF radius both key off the
pixel size, which is why the same parameters need goldens at several of them.
"""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

# 35 mm, upscale 1: pixel_size_um = 35_000 / long_edge.
PIXEL_SIZE_UM = {
    512: 35_000 / 512,
    1024: 35_000 / 1024,
    2048: 35_000 / 2048,
    4000: 35_000 / 4000,
}


def rand_plane() -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(0).random((64, 80)))


def rand_rgb() -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(3).random((48, 60, 3)))


def hdr_rgb() -> np.ndarray:
    """Values up to 4.0, so the boost's protected knee at `0.184 * 2**4 = 2.944` is crossed."""
    return np.ascontiguousarray(np.random.default_rng(7).random((64, 96, 3)) * 4.0)


def step_edge() -> np.ndarray:
    """A hard vertical edge, the fixture most sensitive to the boundary convention."""
    out = np.full((64, 64, 3), 0.01, dtype=np.float64)
    out[:, 32:, :] = 50.0
    return out


def wide_rgb() -> np.ndarray:
    """Large enough that the diffusion-filter radius clamp leaves a 119-tap PSF."""
    return np.ascontiguousarray(np.random.default_rng(21).random((120, 160, 3)))


@fixture
def diffusion_blur():
    """The blur primitive: the FIR kernel, the IIR coefficients, and both paths' output."""
    from spektrafilm.utils.fast_gaussian_filter import (
        _gaussian_kernel_1d,
        _yvv_coeffs,
        fast_gaussian_filter,
    )

    plane = rand_plane()
    yield "blur_plane_64x80", plane
    yield "diffusion_rand_48x60x3", rand_rgb()
    yield "diffusion_hdr_64x96x3", hdr_rgb()
    yield "diffusion_step_edge_64x64x3", step_edge()
    yield "diffusion_wide_120x160x3", wide_rgb()

    for sigma, tag in ((0.7, "0p7"), (1.0, "1p0"), (2.0, "2p0")):
        kernel, _ = _gaussian_kernel_1d(sigma, 3.0)
        yield f"blur_kernel_1d_sigma{tag}", kernel

    coefficient_sigmas = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    yield "blur_yvv_coeffs", np.array([_yvv_coeffs(s) for s in coefficient_sigmas])

    # FIR side of the dispatch. 0.032 gives radius 0, a single unit tap and an exact identity.
    for sigma, tag in ((0.032, "0p032"), (0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0"), (2.99, "2p99")):
        yield f"blur_fir_plane_sigma{tag}", fast_gaussian_filter(plane, sigma)

    # IIR side. Every one of these is 3 to 11 percent wider than the sigma asked for, and uses edge
    # replication rather than the FIR path's reflection.
    for sigma, tag in ((3.0, "3p0"), (5.0, "5p0"), (20.0, "20p0"), (65.0, "65p0")):
        yield f"blur_iir_plane_sigma{tag}", fast_gaussian_filter(plane, sigma)

    # Boundary folds that only a tiny image reaches: 2*radius >= width disables the interior split,
    # and radius > height drives `_reflect` into its modulo branch.
    tiny = np.ascontiguousarray(np.random.default_rng(11).random((3, 11)))
    yield "blur_tiny_3x11", tiny
    yield "blur_fir_tiny_3x11_sigma2p99", fast_gaussian_filter(tiny, 2.99)
    row = np.ascontiguousarray(np.random.default_rng(12).random((1, 40)))
    yield "blur_row_1x40", row
    yield "blur_fir_row_1x40_sigma2p99", fast_gaussian_filter(row, 2.99)

    # Per-channel dispatch inside one call: red and blue FIR, green IIR. These are the third
    # mixture component of the scatter tail at a 4000 px long edge.
    yield (
        "blur_percth_hdr_mixed_dispatch",
        fast_gaussian_filter(hdr_rgb(), np.array([2.9425, 3.0690, 2.8791])),
    )

    # Central row of an IIR impulse response, which is where the excess width is measurable.
    impulse = np.zeros((601, 601), dtype=np.float64)
    impulse[300, 300] = 1.0
    yield (
        "blur_iir_impulse_rows",
        np.stack([fast_gaussian_filter(impulse, s)[300] for s in (3.0, 5.0)]),
    )


@fixture
def diffusion_exponential():
    """The Gaussian-mixture surrogate for the exponential PSF, both fit tables."""
    from spektrafilm.utils.fast_gaussian_filter import fast_exponential_filter

    hdr = hdr_rgb()
    yield "expfilter_n3_hdr_decay5", fast_exponential_filter(hdr, 5.0)
    yield "expfilter_n2_hdr_decay5", fast_exponential_filter(hdr, 5.0, n_gaussians=2)

    # scatter_tail_um / pixel_size_um at a 4000 px long edge. The third component lands at sigma
    # (2.9425, 3.0690, 2.8791), so green alone crosses into the IIR.
    decay = np.array([9.3, 9.7, 9.1]) / PIXEL_SIZE_UM[4000]
    yield "expfilter_n3_hdr_decay_4000px", fast_exponential_filter(hdr, decay)

    yield "expfilter_n3_plane_decay8", fast_exponential_filter(rand_plane(), 8.0)


@fixture
def diffusion_boost():
    """The highlight boost, including the two early-out branches that are not copies."""
    from spektrafilm.utils.numba_boost_hightlights import boost_highlights

    hdr = hdr_rgb()
    yield "boost_hdr_ev3_r0p3_p4", boost_highlights(hdr, 3.0, 0.3, 4.0)
    yield "boost_hdr_ev6_r0p0_p0", boost_highlights(hdr, 6.0, 0.0, 0.0)
    yield "boost_hdr_ev1_r1p0_p2", boost_highlights(hdr, 1.0, 1.0, 2.0)

    # The analytic curve, sampled over ten stops. max_raw is 1024 exactly.
    axis = np.geomspace(1.0e-6, 2.0**10, 512, dtype=np.float64)
    yield (
        "boost_curve_geomspace",
        boost_highlights(np.repeat(axis[:, None, None], 3, axis=2), 10.0, 0.5, 3.0),
    )

    # max_raw == 0 fills zeros instead of copying, which discards the negatives.
    negatives = np.full((2, 3, 3), -1.0, dtype=np.float64)
    negatives[0, 0, 0] = 0.0
    yield "boost_negatives_input", negatives
    yield "boost_negatives_maxzero", boost_highlights(negatives, 2.0, 0.3, 4.0)


@fixture
def diffusion_halation():
    """Scatter and back-reflection, at the pixel sizes where the dispatch changes."""
    from spektrafilm.model.diffusion import apply_halation_um
    from spektrafilm.runtime.params_schema import HalationParams

    hdr = hdr_rgb()
    for edge, tag in ((512, "512"), (1024, "1024"), (4000, "4000")):
        yield (
            f"halation_hdr_default_{tag}px",
            apply_halation_um(hdr, HalationParams(), PIXEL_SIZE_UM[edge]),
        )

    pixel = PIXEL_SIZE_UM[4000]
    yield "halation_step_edge_4000px", apply_halation_um(step_edge(), HalationParams(), pixel)

    # Pass 1 skipped: blue is then bit-identical to the input, because its strength is 0.
    yield (
        "halation_hdr_no_scatter_4000px",
        apply_halation_um(hdr, HalationParams(scatter_amount=0.0), pixel),
    )
    # Pass 2 skipped: every channel's strength is 0, so the `np.any` guard fails.
    yield (
        "halation_hdr_no_bounce_4000px",
        apply_halation_um(hdr, HalationParams(halation_strength=(0.0, 0.0, 0.0)), pixel),
    )
    yield (
        "halation_hdr_no_renorm_4000px",
        apply_halation_um(hdr, HalationParams(halation_renormalize=False), pixel),
    )
    # One bounce, zero decay, off-default scales, strength high enough to see.
    yield (
        "halation_hdr_tuned_4000px",
        apply_halation_um(
            hdr,
            HalationParams(
                scatter_amount=0.6,
                scatter_spatial_scale=2.0,
                halation_amount=1.5,
                halation_spatial_scale=0.5,
                halation_strength=(0.30, 0.10, 0.015),
                halation_first_sigma_um=(50.0, 50.0, 50.0),
                halation_n_bounces=1,
                halation_bounce_decay=0.0,
                halation_renormalize=False,
            ),
            pixel,
        ),
    )


@fixture
def diffusion_filter_tables():
    """The derived PSF tables: group expansion, strength saturation, halo warmth."""
    from spektrafilm.model.diffusion import (
        DIFFUSION_FILTER_FAMILIES,
        _DIFFUSION_FILTER_SHAPES,
        _expand_group,
        _halo_channel_weights,
        _strength_to_scatter,
        diffusion_filter_radial_profile,
    )

    # One flat array in family order, each family contributing core lambdas and weights (2 each),
    # halo (3 each), then bloom (4 each): 18 values per family, 72 in all.
    expanded: list[float] = []
    for family in DIFFUSION_FILTER_FAMILIES:
        shape = _DIFFUSION_FILTER_SHAPES[family]
        for group in ("core", "halo", "bloom"):
            lambdas, weights = _expand_group(shape[group], kind=group)
            expanded.extend(lambdas.tolist())
            expanded.extend(weights.tolist())
    yield "diffusion_expand_tables", np.array(expanded)

    strengths = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 8.0]
    yield (
        "diffusion_strength_scatter",
        np.array([[_strength_to_scatter(s, f) for s in strengths] for f in DIFFUSION_FILTER_FAMILIES]),
    )

    # Halo weights at each family's own warmth base. Cinebloom's 0.85 is the row that clips.
    uniform = np.full(3, 1.0 / 3.0)
    yield (
        "diffusion_halo_weights",
        np.stack(
            [
                _halo_channel_weights(uniform, _DIFFUSION_FILTER_SHAPES[f]["halo_warmth_base"])
                for f in DIFFUSION_FILTER_FAMILIES
            ]
        ),
    )

    radius_um = np.geomspace(1.0, 4000.0, 64, dtype=np.float64)
    yield "diffusion_radial_radius_um", radius_um
    yield (
        "diffusion_radial_profile_bpm",
        diffusion_filter_radial_profile(radius_um, family="black_pro_mist")["total_per_channel"],
    )
    yield (
        "diffusion_radial_profile_cinebloom_overrides",
        diffusion_filter_radial_profile(
            radius_um,
            family="cinebloom",
            spatial_scale=1.5,
            halo_warmth=-0.4,
            overrides={
                "core_intensity": 0.5,
                "halo_intensity": 2.0,
                "bloom_intensity": 1.0,
                "core_size": 1.0,
                "halo_size": 2.0,
                "bloom_size": 0.5,
            },
        )["total_per_channel"],
    )


@fixture
def diffusion_filter_psf():
    """Sampled per-channel PSFs, one per family plus the override quirks."""
    from spektrafilm.model.diffusion import DIFFUSION_FILTER_FAMILIES, diffusion_filter_psf

    yield (
        "psf_bpm_29x29_px100",
        diffusion_filter_psf(
            (29, 29), family="black_pro_mist", spatial_scale=1.0, pixel_size_um=100.0
        ),
    )
    for family in DIFFUSION_FILTER_FAMILIES:
        yield (
            f"psf_{family}_31x31_px150",
            diffusion_filter_psf(
                (31, 31), family=family, spatial_scale=1.0, pixel_size_um=150.0
            ),
        )
    yield (
        "psf_bpm_31x31_warmth_m1p2",
        diffusion_filter_psf(
            (31, 31), family="black_pro_mist", spatial_scale=1.0, pixel_size_um=150.0,
            halo_warmth=-1.2,
        ),
    )
    yield (
        "psf_bpm_31x31_scale2",
        diffusion_filter_psf(
            (31, 31), family="black_pro_mist", spatial_scale=2.0, pixel_size_um=150.0
        ),
    )
    # Negative core intensity clamps to zero and the other two renormalise around it.
    yield (
        "psf_cinebloom_31x31_overrides",
        diffusion_filter_psf(
            (31, 31), family="cinebloom", spatial_scale=1.0, pixel_size_um=150.0,
            halo_warmth=0.3,
            overrides={
                "core_intensity": -5.0,
                "halo_intensity": 1.0,
                "bloom_intensity": 0.25,
                "core_size": 1.5,
                "halo_size": 1.0,
                "bloom_size": 1.0,
            },
        ),
    )
    # All three intensities at zero reverts to the family, sizes and all.
    yield (
        "psf_bpm_31x31_zero_intensities",
        diffusion_filter_psf(
            (31, 31), family="black_pro_mist", spatial_scale=1.0, pixel_size_um=150.0,
            overrides={
                "core_intensity": 0.0,
                "halo_intensity": 0.0,
                "bloom_intensity": 0.0,
                "core_size": 2.0,
                "halo_size": 2.0,
                "bloom_size": 2.0,
            },
        ),
    )


@fixture
def diffusion_filter_apply():
    """The shipping entry point, unclamped and clamped radii, and the mirror boundary."""
    from spektrafilm.model.diffusion import apply_diffusion_filter_um
    from spektrafilm.runtime.params_schema import DiffusionFilterParams

    rgb = rand_rgb()

    def params(**kwargs) -> DiffusionFilterParams:
        return DiffusionFilterParams(active=True, **kwargs)

    # radius 19 at 48x60, under the min(H, W) // 2 - 1 = 23 clamp.
    yield (
        "difffilter_bpm_48x60_px400_s1",
        apply_diffusion_filter_um(rgb, params(filter_family="black_pro_mist", strength=1.0), 400.0),
    )
    # radius 13, also unclamped, and the other end of the strength table.
    yield (
        "difffilter_glimmerglass_48x60_px400_s0p0625",
        apply_diffusion_filter_um(
            rgb, params(filter_family="glimmerglass", strength=0.0625), 400.0
        ),
    )
    # radius wants 77, clamps to 23.
    yield (
        "difffilter_glimmerglass_48x60_px512edge_s2",
        apply_diffusion_filter_um(
            rgb, params(filter_family="glimmerglass", strength=2.0), PIXEL_SIZE_UM[512]
        ),
    )
    yield (
        "difffilter_cinebloom_48x60_px400_warmth",
        apply_diffusion_filter_um(
            rgb,
            params(filter_family="cinebloom", strength=0.75, halo_warmth=0.6, spatial_scale=0.5),
            400.0,
        ),
    )
    yield (
        "difffilter_promist_48x60_px400_overrides",
        apply_diffusion_filter_um(
            rgb,
            params(
                filter_family="pro_mist",
                strength=1.5,
                core_intensity=2.0,
                halo_size=0.5,
                bloom_intensity=0.1,
            ),
            400.0,
        ),
    )
    # A hard edge under a clamped radius of 31, where a wrong boundary fold shows up first.
    yield (
        "difffilter_bpm_step_edge_px200",
        apply_diffusion_filter_um(
            step_edge(), params(filter_family="black_pro_mist", strength=1.0), 200.0
        ),
    )
    # 120x160 with the radius clamped to 59, a 119x119 PSF over a 238x278 padded plane.
    yield (
        "difffilter_bpm_120x160_px100",
        apply_diffusion_filter_um(
            wide_rgb(), params(filter_family="black_pro_mist", strength=0.5), 100.0
        ),
    )


@fixture
def diffusion_lens_blur():
    """Lens blur in both unit systems, and the unsharp mask."""
    from spektrafilm.model.diffusion import (
        apply_gaussian_blur,
        apply_gaussian_blur_um,
        apply_unsharp_mask,
    )

    hdr = hdr_rgb()
    yield "blur_um_hdr_sigma30_4000px", apply_gaussian_blur_um(hdr, 30.0, PIXEL_SIZE_UM[4000])
    yield "blur_px_hdr_sigma0p9", apply_gaussian_blur(hdr, 0.9)
    yield "unsharp_plane_64x80_s0p7_a0p7", apply_unsharp_mask(rand_plane(), 0.7, 0.7)
    yield "unsharp_rgb_48x60_s1p5_a1p2", apply_unsharp_mask(rand_rgb(), 1.5, 1.2)
