"""Goldens for the grain particle model and for glare.

Three kinds live here.

Deterministic: every table `apply_grain` derives before the first random draw, the elementwise `p`
and `sat` planes, and the sublayer split. Gated at the project tolerance or tighter.

Closed form: the mean, standard deviation, skewness and excess kurtosis the particle model implies
(`grain.md` section 4.2), so the Swift side can check its own derived-parameter chain against the
oracle exactly and then gate its samples against the closed form. Also the sampling standard
deviation of each statistic, measured across independent realisations at the sample count the Swift
tests use, so the 5-sigma gates are calibrated from data.

Sampled: the oracle's own realisation of the layered path and of the RMS granularity figure. These
are one draw from a different RNG, so they are regression references. The tests gate them against
the same closed form as the Swift samples.
"""

from __future__ import annotations

import numpy as np

from fixture_registry import fixture

STOCK = "kodak_portra_400"
POSITIVE_STOCK = "fujifilm_velvia_100"

# 35 mm at 4000 px wide. Every derived-parameter number in grain.md section 9.1 is at this pitch.
PIXEL_SIZE_UM = 8.75

# A 48 um densitometer aperture has area (48/2)^2 * pi, and utils/plotting.grain_test renders at the
# pixel size with that area so `std * 1000` is the industry RMS granularity figure.
DENSITOMETER_PIXEL_SIZE_UM = float(np.sqrt((48 / 2) ** 2 * np.pi))

# Low density through saturation: 0.05 is a few particles and strongly skewed, 1.9 is within 0.1 of
# green's total density_max and clips there.
DENSITY_LEVELS = [0.05, 0.2, 0.5, 1.0, 1.5, 1.9]

# Levels the single-layer path is gated at. It is not the production topology, so it gets only the
# ends and the middle of the sweep, to keep the Swift suite's runtime down.
SINGLE_LAYER_LEVELS = [0.05, 0.5, 1.9]

# Sample count the Swift statistical gates use, and the realisation count the sampling standard
# deviations are measured over.
SAMPLES = 512 * 512
REALISATIONS = 48


def _grain_params(**overrides):
    from spektrafilm.runtime.params_schema import GrainParams

    return GrainParams(**overrides)


def _quiet_params(**overrides):
    """Defaults with every spatial and clumping stage off, so the samples are i.i.d."""
    base = dict(blur=0.0, blur_dye_clouds_um=0.0, micro_structure=(0.0, 0.0))
    base.update(overrides)
    return _grain_params(**base)


def _curves(stock):
    from spektrafilm.profiles.io import load_profile

    profile = load_profile(stock)
    curves = np.asarray(profile.data.density_curves)
    layers = np.asarray(profile.data.density_curves_layers)
    normalised = curves - np.nanmin(curves, axis=0)
    return profile, normalised, layers


def _single_layer_tables(grain, density_max_curves, pixel_size_um):
    density_min = np.asarray(grain.density_min)
    particle_area = grain.particle_area_um2 * np.asarray(grain.particle_scale)
    particles = pixel_size_um**2 / particle_area
    if grain.n_sub_layers > 1:
        particles = particles / grain.n_sub_layers
    return np.stack(
        [
            np.asarray(density_max_curves),
            density_min,
            density_max_curves + density_min,
            particle_area,
            particles,
        ]
    )


def _layered_tables(grain, density_max_layers, pixel_size_um):
    total = np.sum(density_max_layers, axis=0)
    fractions = density_max_layers / total[None, :]
    min_layers = fractions * np.asarray(grain.density_min)[None, :]
    max_layers = density_max_layers + min_layers
    area = (
        grain.particle_area_um2
        * np.asarray(grain.particle_scale)[None, :]
        * np.asarray(grain.particle_scale_layers)[:, None]
    )
    particles = pixel_size_um**2 * fractions / area
    od = max_layers / particles
    dye_sigma = grain.blur_dye_clouds_um * np.sqrt(od)
    tables = np.stack(
        [density_max_layers, fractions, min_layers, max_layers, area, particles, od, dye_sigma]
    )
    return tables, total, fractions, min_layers, max_layers, particles, od


def _population(density, density_max, particles, uniformity, weight):
    """One scaled Poisson: `weight * od * sat * Poisson(N * p / sat)`."""
    p = np.clip(density / density_max, 1e-6, 1 - 1e-6)
    sat = 1 - p * uniformity * (1 - 1e-6)
    lam = particles * p / sat
    step = weight * density_max / particles * sat
    return step, lam


def _moments(steps, lambdas, offset):
    """Mean, sd, skewness and excess kurtosis of a sum of independent scaled Poissons."""
    steps = np.asarray(steps, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    cumulants = [np.sum(steps**k * lambdas) for k in (1, 2, 3, 4)]
    variance = cumulants[1]
    return np.array(
        [
            cumulants[0] + offset,
            np.sqrt(variance),
            cumulants[2] / variance**1.5,
            cumulants[3] / variance**2,
        ]
    )


def _layered_populations(grain, density, stock=STOCK, pixel_size_um=PIXEL_SIZE_UM):
    """Per-channel (steps, lambdas) for a flat input density on the layered path."""
    from spektrafilm.model.density_curves import interp_density_cmy_layers

    profile, normalised, layers = _curves(stock)
    density_max_layers = np.nanmax(layers, axis=0)
    _, _, _, min_layers, max_layers, particles, _ = _layered_tables(
        grain, density_max_layers, pixel_size_um
    )
    split = interp_density_cmy_layers(
        np.full((1, 1, 3), float(density)),
        normalised,
        layers,
        positive_film=profile.info.type == "positive",
    )[0, 0]  # [sublayer, channel]

    out = []
    for channel in range(3):
        steps, lambdas = [], []
        for sublayer in range(3):
            step, lam = _population(
                split[sublayer, channel] + min_layers[sublayer, channel],
                max_layers[sublayer, channel],
                particles[sublayer, channel],
                grain.uniformity[channel],
                1.0,
            )
            steps.append(step)
            lambdas.append(lam)
        out.append((np.array(steps), np.array(lambdas)))
    return out


def _single_layer_populations(density, repeats, stock=STOCK, pixel_size_um=PIXEL_SIZE_UM):
    """Per-channel (steps, lambdas) for a flat input density on the single-layer path."""
    grain = _quiet_params(sublayers_active=False, n_sub_layers=repeats)
    _, normalised, _ = _curves(stock)
    tables = _single_layer_tables(grain, np.nanmax(normalised, axis=0), pixel_size_um)
    density_max, particles = tables[2], tables[4]

    out = []
    for channel in range(3):
        step, lam = _population(
            density + grain.density_min[channel],
            density_max[channel],
            particles[channel],
            grain.uniformity[channel],
            1.0 / repeats,
        )
        out.append((np.full(repeats, step), np.full(repeats, lam)))
    return out


@fixture
def grain_derived():
    """Every table the two topologies derive before the first random draw."""
    _, normalised, layers = _curves(STOCK)
    density_max_curves = np.nanmax(normalised, axis=0)
    density_max_layers = np.nanmax(layers, axis=0)

    grain = _grain_params()
    yield (
        f"grain_derived_single_{STOCK}",
        _single_layer_tables(grain, density_max_curves, PIXEL_SIZE_UM),
    )
    # n_sub_layers > 1 is the only branch in the single-layer derivation.
    yield (
        f"grain_derived_single_sub3_{STOCK}",
        _single_layer_tables(
            _grain_params(n_sub_layers=3), density_max_curves, PIXEL_SIZE_UM
        ),
    )

    tables, total, *_ = _layered_tables(grain, density_max_layers, PIXEL_SIZE_UM)
    yield f"grain_derived_layers_{STOCK}", tables
    yield f"grain_derived_layers_total_{STOCK}", total

    # The micro-structure gates as a function of pixel pitch. Both are false at every pitch a real
    # render uses; the last row is the forced-on fixture from grain.md section 9.6.
    rows = []
    for micro in [(0.2, 30.0), (0.2, 300.0)]:
        for pixel in [42.5388924217, 8.75, 5.8333333333, 2.9166666667, 0.6, 0.5, 0.3]:
            rows.append([pixel, micro[0], micro[1], micro[0] / pixel, micro[1] * 0.001 / pixel])
    yield "grain_derived_micro_gates", np.asarray(rows)


@fixture
def grain_model():
    """The elementwise pieces, the sublayer split, and the closed-form moments."""
    from spektrafilm.model.density_curves import interp_density_cmy_layers

    # Negative, zero and above density_max all have to be covered: the clip on `p` floors the first
    # two to 1e-6 and saturates the last at 1 - 1e-6.
    rng = np.random.default_rng(20250922)
    plane = rng.uniform(-0.6, 2.6, size=(8, 9))
    plane[0, :3] = [-0.5, -0.01, 0.0]
    plane[0, 3:6] = [1e-9, 0.5, 2.2]
    plane[0, 6:9] = [2.3, 5.0, np.nan]
    yield "grain_density_plane_input", plane

    cases = np.array(
        [
            [2.2, 0.97],
            [1.8, 0.99],
            [0.605979294827, 0.97],  # Portra 400 sublayer 0, red, with density_min folded in
            [2.2, 1.0],  # uniformity 1, where sat bottoms out at 2e-6 and lambda peaks
        ]
    )
    yield "grain_p_sat_params", cases
    stacked = []
    for density_max, uniformity in cases:
        p = np.clip(plane / density_max, 1e-6, 1 - 1e-6)
        stacked.append(np.stack([p, 1 - p * uniformity * (1 - 1e-6)]))
    yield "grain_p_sat", np.stack(stacked)

    # The sublayer split, on a ramp that leaves the curve range at both ends.
    ramp = np.linspace(-0.4, 2.8, 6 * 7 * 3).reshape(6, 7, 3)
    yield "grain_sublayer_input", ramp
    for stock in [STOCK, POSITIVE_STOCK]:
        profile, normalised, layers = _curves(stock)
        yield (
            f"grain_sublayer_split_{stock}",
            interp_density_cmy_layers(
                np.ascontiguousarray(ramp),
                normalised,
                layers,
                positive_film=profile.info.type == "positive",
            ),
        )

    yield "grain_closed_form_levels", np.asarray(DENSITY_LEVELS)
    yield "grain_single_layer_levels", np.asarray(SINGLE_LAYER_LEVELS)

    grain = _quiet_params()
    layered = np.zeros((len(DENSITY_LEVELS), 3, 4))
    for row, density in enumerate(DENSITY_LEVELS):
        for channel, (steps, lambdas) in enumerate(_layered_populations(grain, density)):
            layered[row, channel] = _moments(steps, lambdas, -grain.density_min[channel])
    yield f"grain_closed_form_layers_{STOCK}", layered

    _, normalised, _ = _curves(STOCK)
    density_max_curves = np.nanmax(normalised, axis=0)
    for repeats in (1, 3):
        out = np.zeros((len(SINGLE_LAYER_LEVELS), 3, 4))
        for row, density in enumerate(SINGLE_LAYER_LEVELS):
            for channel, (steps, lambdas) in enumerate(
                _single_layer_populations(density, repeats)
            ):
                out[row, channel] = _moments(steps, lambdas, -_quiet_params().density_min[channel])
        yield f"grain_closed_form_single_sub{repeats}_{STOCK}", out


@fixture
def grain_statistics():
    """Sampling standard deviations, and the oracle's own realisation of the layered path."""
    from scipy.stats import skew

    from spektrafilm.model.grain import apply_grain

    grain = _quiet_params()
    profile, normalised, layers = _curves(STOCK)

    rng = np.random.default_rng(777)

    def sampling_sd(levels, populations_for):
        """SD of the mean, the sd and the skewness across independent realisations.

        Drawn straight from `Poisson` rather than through `apply_grain`, so it measures the
        estimator's noise at SAMPLES samples and not the reference's sampler. The Swift
        gates are five of these, so they are calibrated rather than assumed normal: `sqrt(6/n)`
        understates the skewness estimator's spread by up to 25 percent here.
        """
        out = np.zeros((len(levels), 3, 3))
        for row, density in enumerate(levels):
            for channel, (steps, lambdas) in enumerate(populations_for(density)):
                stats = np.zeros((REALISATIONS, 3))
                for r in range(REALISATIONS):
                    total = np.zeros(SAMPLES)
                    for step, lam in zip(steps, lambdas):
                        total += step * rng.poisson(lam, SAMPLES)
                    stats[r] = [np.mean(total), np.std(total), skew(total)]
                out[row, channel] = np.std(stats, axis=0)
        return out

    yield (
        f"grain_layered_moment_sd_{STOCK}",
        sampling_sd(DENSITY_LEVELS, lambda d: _layered_populations(grain, d)),
    )
    for repeats in (1, 3):
        yield (
            f"grain_single_moment_sd_sub{repeats}_{STOCK}",
            sampling_sd(
                SINGLE_LAYER_LEVELS, lambda d, r=repeats: _single_layer_populations(d, r)
            ),
        )

    measured = np.zeros((len(DENSITY_LEVELS), 3, 3))
    for row, density in enumerate(DENSITY_LEVELS):
        out = apply_grain(
            np.full((512, 512, 3), float(density)),
            PIXEL_SIZE_UM,
            grain,
            normalised,
            layers,
            profile.info.type,
        )
        for channel in range(3):
            flat = out[:, :, channel].ravel()
            measured[row, channel] = [np.mean(flat), np.std(flat), skew(flat)]
    yield f"grain_oracle_layered_moments_{STOCK}", measured

    for repeats in (1, 3):
        single = _quiet_params(sublayers_active=False, n_sub_layers=repeats)
        out = np.zeros((len(SINGLE_LAYER_LEVELS), 3, 3))
        for row, density in enumerate(SINGLE_LAYER_LEVELS):
            # apply_grain_to_density adds density_min into its argument in place, so pass a copy.
            grained = apply_grain(
                np.full((512, 512, 3), float(density)),
                PIXEL_SIZE_UM,
                single,
                normalised,
                layers,
                profile.info.type,
            )
            for channel in range(3):
                flat = grained[:, :, channel].ravel()
                out[row, channel] = [np.mean(flat), np.std(flat), skew(flat)]
        yield f"grain_oracle_single_moments_sub{repeats}_{STOCK}", out

    # RMS granularity through a 48 um aperture, the figure a photographer would recognise. Twice:
    # with the spatial stages off, and with production defaults on an interior crop.
    for label, params, crop in (
        ("quiet", grain, 0),
        ("default", _grain_params(), 8),
    ):
        rms = np.zeros((len(DENSITY_LEVELS), 3))
        for row, density in enumerate(DENSITY_LEVELS):
            out = apply_grain(
                np.full((512, 512, 3), float(density)),
                DENSITOMETER_PIXEL_SIZE_UM,
                params,
                normalised,
                layers,
                profile.info.type,
            )
            interior = out[crop : out.shape[0] - crop, crop : out.shape[1] - crop] if crop else out
            rms[row] = np.std(interior.reshape(-1, 3), axis=0) * 1000
        yield f"grain_rms_granularity_{label}_{STOCK}", rms


@fixture
def grain_clumping_and_glare():
    """The unit-mean lognormal field both the clumping stage and glare are built from.

    These moments are the CLOSED FORM, not a measurement of the reference.

    `fast_lognormal_from_mean_std` draws through `np.random.randn()` inside `njit(parallel=True)`, so
    Numba keeps per-thread generator state and the thread schedule decides which thread draws which
    value. Seeding does not make it reproducible: two consecutive regenerations of a sampled version
    differed by 3.9% relative for glare and 0.25% for the clumping field. A sampled fixture would
    dirty the tree on every `make goldens` run and defeat the "review the delta" rule.

    The distribution's mean and standard deviation are known exactly by construction, so the fixture
    stores those. They are reproducible, and they test that the Swift sampler has the right
    distribution, not that it agrees with one noisy draw of the Python one. The Swift side compares
    its own sample against these within its measured sampling error.
    """
    rng = np.random.default_rng(4242)

    # Mean 1 and standard deviation `sigma`, by the definition of the parameterisation.
    rows = [[sigma, 1.0, sigma] for sigma in (1.0, 0.3, 0.05001)]
    yield "grain_clumping_moments", np.asarray(rows)

    # Glare, after the division by 100. Blur is 0 in every row so the samples stay i.i.d.; the
    # blurred case is gated against the kernel's variance retention instead of a stored number.
    params = np.array([[0.03, 0.7], [0.03, 0.0], [1.0, 0.25], [5.0, 1.5]])
    yield "glare_moments_params", params
    moments = np.zeros((len(params), 2))
    sampling_sd = np.zeros((len(params), 2))
    for row, (percent, roughness) in enumerate(params):
        # Closed form again, after the division by 100 the glare model applies.
        moments[row] = [percent / 100, roughness * percent / 100]

        # Sampling noise of the two statistics, from the same lognormal drawn REALISATIONS times.
        mean, std = percent / 100, roughness * percent / 100
        if std > 0:
            sigma2 = np.log1p(std**2 / mean**2)
            mu = np.log(mean) - sigma2 / 2
            draws = np.exp(
                mu + np.sqrt(sigma2) * rng.standard_normal((REALISATIONS, SAMPLES))
            )
            sampling_sd[row] = [np.std(draws.mean(axis=1)), np.std(draws.std(axis=1))]
        else:
            sampling_sd[row] = [0.0, 0.0]
    yield "glare_moments", moments
    yield "glare_moment_sd", sampling_sd
