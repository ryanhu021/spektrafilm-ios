"""Goldens for the emulsion model: density curves, DIR couplers and the spectral products."""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

# Two negatives and two positives. Velvia and Provia exercise the positive branch of the coupler
# inversion, where the shifted exposure axis runs backwards.
STOCKS = ["kodak_portra_400", "kodak_vision3_500t", "fujifilm_velvia_100", "kodak_ektachrome_100"]


@fixture
def couplers():
    """The inhibition matrix and the curves it implies before the couplers acted."""
    from spektrafilm.model.couplers import (
        compute_density_curves_before_dir_couplers,
        compute_dir_couplers_matrix,
    )
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.params_schema import DirCouplersParams

    yield "couplers_matrix_default", compute_dir_couplers_matrix(DirCouplersParams())

    # A non-default parameter set, so the test covers the scaling paths as well as the defaults.
    tuned = DirCouplersParams(
        inhibition_samelayer=0.6,
        inhibition_interlayer=1.4,
        gamma_samelayer_rgb=(0.4, 0.3, 0.2),
        gamma_interlayer_r_to_gb=(0.2, 0.1),
        gamma_interlayer_g_to_rb=(0.3, 0.15),
        gamma_interlayer_b_to_rg=(0.25, 0.35),
    )
    yield "couplers_matrix_tuned", compute_dir_couplers_matrix(tuned)

    for stock in STOCKS:
        profile = load_profile(stock)
        curves = np.asarray(profile.data.density_curves)
        log_exposure = np.asarray(profile.data.log_exposure)
        normalised = curves - np.nanmin(curves, axis=0)
        positive = profile.info.type == "positive"

        for label, params in (("default", DirCouplersParams()), ("tuned", tuned)):
            matrix = compute_dir_couplers_matrix(params) * params.amount
            yield (
                f"couplers_before_{stock}_{label}",
                compute_density_curves_before_dir_couplers(
                    normalised, log_exposure, matrix, positive=positive
                ),
            )


@fixture
def density_curves():
    """Exposure to density, and the spectral products that follow."""
    from opt_einsum import contract

    from spektrafilm.config import STANDARD_OBSERVER_CMFS
    from spektrafilm.model.density_curves import interpolate_exposure_to_density
    from spektrafilm.model.develop import compute_density_spectral
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.utils.conversions import density_to_light

    rng = np.random.default_rng(1104)
    # Spans past both ends of the exposure axis so the clamping is covered.
    log_raw = np.ascontiguousarray(rng.uniform(-4.5, 5.5, size=(12, 10, 3)))
    yield "density_log_raw_input", log_raw

    for stock in STOCKS:
        profile = load_profile(stock)
        curves = np.asarray(profile.data.density_curves)
        log_exposure = np.asarray(profile.data.log_exposure)
        normalised = curves - np.nanmin(curves, axis=0)

        yield f"density_curves_normalised_{stock}", normalised
        yield (
            f"density_from_exposure_{stock}_gamma1",
            interpolate_exposure_to_density(log_raw, normalised, log_exposure, 1.0),
        )
        yield (
            f"density_from_exposure_{stock}_gamma_rgb",
            interpolate_exposure_to_density(
                log_raw, normalised, log_exposure, [0.85, 1.0, 1.2]
            ),
        )

    # Spectral products, on Portra 400 only. These are per-pixel maps, so one stock exercises the
    # arithmetic and the NaN handling that comes with partial datasheet coverage.
    profile = load_profile("kodak_portra_400")
    channel_density = np.asarray(profile.data.channel_density)
    base_density = np.asarray(profile.data.base_density)
    density_cmy = interpolate_exposure_to_density(
        log_raw, np.asarray(profile.data.density_curves), np.asarray(profile.data.log_exposure), 1.0
    )
    yield "spectral_density_cmy_input", density_cmy
    yield "spectral_channel_density", channel_density
    yield "spectral_base_density", base_density

    with_base = compute_density_spectral(channel_density, density_cmy, base_density=base_density)
    without_base = compute_density_spectral(channel_density, density_cmy, base_density=None)
    yield "spectral_density_with_base", with_base
    yield "spectral_density_without_base", without_base

    illuminant = standard_illuminant("D50")
    light = density_to_light(with_base, illuminant)
    yield "spectral_light_d50", light

    normalisation = np.sum(illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
    xyz = contract("ijk,kl->ijl", light, STANDARD_OBSERVER_CMFS[:]) / normalisation
    yield "spectral_xyz_d50", xyz


@fixture
def enlarger():
    """The colour enlarger head: dichroic filtering in Kodak CC units."""
    from spektrafilm.model.color_filters import (
        color_enlarger,
        custom_dichroic_filters,
        durst_digital_light_dicrhoic_filters,
        edmund_optics_dichroic_filters,
        thorlabs_dichroic_filters,
    )
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.runtime.services.filter_enlarger_source import EnlargerService
    from spektrafilm.runtime.params_schema import EnlargerParams

    lamp = standard_illuminant("TH-KG3")
    yield "enlarger_lamp_th_kg3", lamp

    # Default neutral positions, then two off-neutral settings.
    for label, cc in (
        ("neutral", (0.0, 65.0, 55.0)),
        ("open", (0.0, 0.0, 0.0)),
        ("heavy", (20.0, 90.0, 80.0)),
    ):
        yield f"enlarger_cc_{label}", color_enlarger(lamp, filter_cc_values=cc)

    # Every filter set, at the default neutral position, so a wrong table is caught.
    for name, filters in (
        ("custom", custom_dichroic_filters),
        ("thorlabs", thorlabs_dichroic_filters),
        ("edmund", edmund_optics_dichroic_filters),
        ("durst", durst_digital_light_dicrhoic_filters),
    ):
        yield (
            f"enlarger_set_{name}",
            color_enlarger(lamp, filter_cc_values=(0.0, 65.0, 55.0), filters=filters),
        )

    # The service's three accessors, with shifts engaged so neutral and filtered differ.
    params = EnlargerParams(
        m_filter_shift=12.0,
        y_filter_shift=-8.0,
        preflash_m_filter_shift=5.0,
        preflash_y_filter_shift=3.0,
    )
    service = EnlargerService(params)
    yield "enlarger_service_filtered", service.enlarger_filtered_illuminant(lamp)
    yield "enlarger_service_neutral", service.enlarger_neutral_illuminant(lamp)
    yield "enlarger_service_preflash", service.preflash_filtered_illuminant(lamp)


@fixture
def autoexposure():
    """All seven metering methods, plus the luminance they all start from."""
    from spektrafilm.utils.autoexposure import _luminance_y, measure_autoexposure_ev

    rng = np.random.default_rng(88123)
    # Non-square, so the long-edge normalisation of the coordinate grid matters, and large enough
    # that the 5x5 matrix grid has full cells.
    image = np.ascontiguousarray(rng.uniform(0.0, 1.4, size=(37, 61, 3)))
    yield "autoexposure_input", image

    for space, decode in (("sRGB", False), ("ProPhoto RGB", False), ("sRGB", True)):
        slug = space.lower().replace(" ", "_") + ("_decoded" if decode else "")
        yield f"autoexposure_luminance_{slug}", _luminance_y(image, space, decode)

    methods = [
        "average", "median", "center_weighted", "partial",
        "matrix", "multi_zone", "highlight_weighted",
    ]
    yield (
        "autoexposure_ev",
        np.array([
            measure_autoexposure_ev(image, "ProPhoto RGB", False, method=m) for m in methods
        ]),
    )

    # A fully black frame, where the reference guards against a -inf result.
    black = np.zeros((8, 8, 3))
    yield "autoexposure_ev_black", np.array([
        measure_autoexposure_ev(black, "ProPhoto RGB", False, method=m) for m in methods
    ])


# Per-stage taps, and the end-to-end render. These are the fixtures that say the stages are wired
# in the right order, as opposed to each being individually correct.
TAPS = ["rgb_pre", "log_e_film", "cmy_film", "log_e_print", "cmy_print", "rgb_out"]

# lut_mode makes the pipeline a deterministic per-pixel transform, so these fixtures are stable and
# do not depend on the RNG. The spatial and stochastic stages get their own gates elsewhere.
TAP_CASES = [
    ("portra400_endura_srgb", "kodak_portra_400", "kodak_portra_endura", "sRGB"),
    ("velvia_endura_srgb", "fujifilm_velvia_100", "kodak_portra_endura", "sRGB"),
    ("vision3_500t_2383_srgb", "kodak_vision3_500t", "kodak_2383", "sRGB"),
    ("portra400_endura_p3", "kodak_portra_400", "kodak_portra_endura", "Display P3"),
]


@fixture
def pipeline_taps():
    """Every tap for four film, paper and output-space combinations, plus a grey ramp end to end."""
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    # A patch per distinct input value, so one fixture covers shadows through highlights without
    # needing a real photograph in git.
    values = [0.02, 0.09, 0.184, 0.4, 0.9, 2.0]
    ramp = np.zeros((2, 3, 3))
    for i, v in enumerate(values):
        ramp[i // 3, i % 3, :] = v
    yield "pipeline_ramp_input", ramp

    for label, film, paper, output_space in TAP_CASES:
        params = init_params(film_profile=film, print_profile=paper)
        params.camera.auto_exposure = False
        params.debug.lut_mode = True
        params.io.output_color_space = output_space
        params = digest_params(params)

        for tap in TAPS:
            pipeline = SimulationPipeline(params)
            yield f"pipeline_{label}_{tap}", pipeline.process(ramp, collect=tap)

    # Scanning the negative directly, which takes a different topology.
    params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
    params.camera.auto_exposure = False
    params.debug.lut_mode = True
    params.io.scan_film = True
    params = digest_params(params)
    for tap in ["rgb_pre", "log_e_film", "cmy_film", "rgb_out"]:
        pipeline = SimulationPipeline(params)
        yield f"pipeline_scanfilm_portra400_{tap}", pipeline.process(ramp, collect=tap)


@fixture
def develop_full():
    """The full develop() composition, which the per-function fixtures do not cover.

    Grain off, so this isolates the density curves plus the DIR coupler application. The coupler
    spatial term is off via diffusion_size_um = 0, which is what deactivate_spatial_effects sets;
    passing pixel_size_um = None instead raises, because the reference divides by it before checking.
    """
    from spektrafilm.model.couplers import apply_density_correction_dir_couplers
    from spektrafilm.model.density_curves import interpolate_exposure_to_density
    from spektrafilm.model.develop import develop
    from spektrafilm.profiles.io import load_profile
    from spektrafilm.runtime.params_schema import DirCouplersParams, GrainParams

    log_raw = np.ascontiguousarray(
        np.random.default_rng(4242).uniform(-4.0, 4.5, size=(8, 8, 3)))
    yield "develop_log_raw_input", log_raw

    grain_off = GrainParams(active=False)
    for stock in ["kodak_portra_400", "fujifilm_velvia_100"]:
        profile = load_profile(stock)
        curves = np.asarray(profile.data.density_curves)
        log_exposure = np.asarray(profile.data.log_exposure)
        normalised = curves - np.nanmin(curves, axis=0)
        couplers = DirCouplersParams(diffusion_size_um=0.0)

        # The intermediate, before the couplers act.
        before = interpolate_exposure_to_density(log_raw, normalised, log_exposure, 1.0)
        yield f"develop_{stock}_precoupler", before

        # The coupler correction on its own, which is the step with no fixture until now.
        yield (
            f"develop_{stock}_coupled",
            apply_density_correction_dir_couplers(
                before, log_raw, 10.0, log_exposure, normalised, couplers,
                profile.info.type, gamma_factor=1.0,
            ),
        )

        # And the whole function, grain off.
        yield (
            f"develop_{stock}_full",
            develop(
                log_raw, 10.0, log_exposure, curves,
                np.asarray(profile.data.density_curves_layers),
                couplers, grain_off, profile.info.type, gamma_factor=1.0,
            ),
        )


@fixture
def print_balance():
    """The midgray reference and the exposure factor the print balance derives from."""
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
    params.camera.auto_exposure = False
    params.debug.lut_mode = True
    params = digest_params(params)

    pipeline = SimulationPipeline(params)
    midgray = pipeline._enlarger_service.density_spectral_midgray
    yield "print_balance_midgray_spectral", midgray
    yield (
        "print_balance_midgray_comp_is_none",
        np.array([1.0 if pipeline._enlarger_service.density_spectral_midgray_comp is None else 0.0]),
    )

    # The paper sensitivity and the filtered lamp, so the Swift side can be checked at each step.
    from spektrafilm.model.illuminants import standard_illuminant
    lamp = standard_illuminant(params.enlarger.illuminant)
    filtered = pipeline._enlarger_service.enlarger_filtered_illuminant(lamp)
    yield "print_balance_filtered_illuminant", filtered

    sensitivity = np.nan_to_num(10 ** np.asarray(params.print.data.log_sensitivity))
    yield "print_balance_paper_sensitivity", sensitivity

    from spektrafilm.runtime.stages.printing import _exposure_factor
    yield "print_balance_exposure_factor", _exposure_factor(sensitivity, filtered, midgray)

    yield "print_balance_neutral_cmy", np.array([
        params.enlarger.c_filter_neutral,
        params.enlarger.m_filter_neutral,
        params.enlarger.y_filter_neutral,
    ])


@fixture
def midgray_steps():
    """Each step of _simple_rgb_to_density_spectral, so a divergence can be localised."""
    from spektrafilm.model.develop import develop_simple
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
    params.camera.auto_exposure = False
    params.debug.lut_mode = True
    params = digest_params(params)
    pipeline = SimulationPipeline(params)
    filming = pipeline._filming_stage

    rgb = np.array([[[0.184] * 3]])
    raw = filming._rgb_to_film_raw(rgb)
    yield "midgray_raw", raw
    log_raw = np.log10(raw + 1e-10)
    yield "midgray_log_raw", log_raw
    yield (
        "midgray_cmy",
        develop_simple(
            log_raw,
            np.asarray(params.film.data.log_exposure),
            np.asarray(params.film.data.density_curves),
            gamma_factor=params.film_render.density_curve_gamma,
        ),
    )

    # The same raw through the expose path's own log, for comparison.
    yield "midgray_log_raw_fmax", np.log10(np.fmax(raw, 0.0) + 1e-10)


@fixture
def resample():
    """skimage rescale at order 0, which auto-exposure's 256 px preview depends on."""
    from skimage.transform import rescale

    rng = np.random.default_rng(31337)
    # Small deliberately: these live in git, and the arithmetic does not depend on the size.
    for label, h, w in [("80x53", 53, 80), ("odd_53x97", 53, 97), ("square_64", 64, 64)]:
        image = np.ascontiguousarray(rng.uniform(0.0, 1.5, size=(h, w, 3)))
        yield f"resample_input_{label}", image
        for factor in [0.4, 0.213, 0.5, 0.77]:
            tag = str(factor).replace(".", "p")
            yield (
                f"resample_order0_{label}_{tag}",
                rescale(image, factor, channel_axis=2, order=0),
            )

    # The exact call ResizingService.small_preview makes on a 640 px preview.
    # The same 0.4 factor small_preview hits on a 640 px preview, at a size git can hold.
    preview = np.ascontiguousarray(rng.uniform(0.0, 1.2, size=(107, 160, 3)))
    yield "resample_preview_input", preview
    yield "resample_preview_256", rescale(preview, 64 / 160, channel_axis=2, order=0)

    # And the metered EV that comes out of it, which is what actually reaches the render.
    from spektrafilm.utils.autoexposure import measure_autoexposure_ev
    small = rescale(preview, 64 / 160, channel_axis=2, order=0)
    yield "resample_preview_ev", np.array([
        measure_autoexposure_ev(small, "Display P3", False, method="center_weighted")
    ])
