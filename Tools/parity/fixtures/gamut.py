"""Goldens for input and output gamut compression.

Covers the Reinhard knee, the spectral locus geometry, the four perceptual transforms, the C_max
chroma envelopes and all five output compressors. The pixels run well outside the target gamut and
above white, because the simulation feeds this stage such values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fixture_registry import fixture, sidecar

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "Tests" / "SpektraFilmTests" / "Goldens"

WHITE_E = (1.0 / 3.0, 1.0 / 3.0)
# kodak_portra_400's reference illuminant, the most common one across the bundled profiles.
WHITE_D55 = (0.3324316389215494, 0.3474443989306461)

DEFAULT_KNEE = (0.0, 1.0, 6.0)
# The two knees the reference's own test file uses for the perceptual algorithms.
SOFT_KNEE = (0.815, 1.0, 1.2)
LATE_KNEE = (0.95, 1.0, 2.0)

KNEE_SWEEP = np.concatenate([
    np.array([-1e9, -0.3, -1e-9, 0.0, 1e-9]),
    np.linspace(0.0, 3.0, 121),
    np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 1e3, 1e9]),
])

# Eight chromaticities: white, just off white, two near the red and blue ends of the locus, the sRGB
# and BT.2020 red primaries, the origin and a high-y point.
XY_CASES = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [0.35, 0.36],
    [0.73, 0.28],
    [0.15, 0.06],
    [0.64, 0.33],
    [0.708, 0.292],
    [0.0, 0.0],
    [0.1, 0.8],
])

RGB_CASES = np.array([
    [0.5, 0.5, 0.5],
    [1.0, 0.3, 0.3],
    [1.5, -0.1, -0.05],
    [-0.2, 1.0, 1.0],
    [1.2, -0.05, -0.05],
    [0.0, 0.0, 0.0],
    [2.0, -0.1, 0.3],
    [0.05, 0.02, 0.4],
])

PERCEPTUAL = ("oklch", "oklrab", "jzazbz", "cam16ucs")

# Non-finite pixels and chromaticities. Nothing upstream should produce them. They are pinned
# because the reference does not simply propagate them: colour-science's sdiv turns a NaN quotient
# into 0 and hue_quadrature overwrites a NaN hue angle with 0, so several of these come back as
# numbers on the reference side. A port that propagates NaN instead disagrees here.
# The three NaN rows have distinct finite channels so the aces_rgc path can tell a NaN-propagating
# maximum from one that drops it: np.max propagates, and the pixel then passes through untouched.
NONFINITE_RGB = np.array([
    [np.nan, 0.3, 0.7],
    [0.9, np.nan, 0.1],
    [0.8, 0.2, np.nan],
    [np.inf, 0.2, 0.2],
    [-np.inf, 0.2, 0.2],
    [0.4, 0.5, 0.6],
])

NONFINITE_XY = np.array([
    [np.nan, 0.4],
    [0.4, np.nan],
    [np.inf, 0.3],
    [0.35, 0.36],
])


def _xy_grid(n: int = 40) -> np.ndarray:
    """A square of chromaticities spanning the locus and a margin outside it."""
    axis = np.linspace(-0.05, 0.95, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], axis=-1)


def xy_input() -> np.ndarray:
    return np.concatenate([XY_CASES, _xy_grid()], axis=0)


def rgb_input() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([RGB_CASES, rng.uniform(-0.2, 1.3, size=(1024, 3))], axis=0)


def xyz_input() -> np.ndarray:
    """XYZ samples for the transform round trips, including a few the pipeline cannot produce."""
    rng = np.random.default_rng(0)
    edge = np.array([
        [0.9504559270516716, 1.0, 1.0890577507598784],  # D65 white at Y = 1
        [0.0, 0.0, 0.0],
        [1e-9, 1e-9, 1e-9],
        [0.2, 0.05, 1.4],
        [1.6, 1.6, 1.6],
        [0.4, 0.0, 0.0],
    ])
    return np.concatenate([edge, rng.uniform(0.001, 1.6, size=(512, 3))], axis=0)


def realizable_input() -> np.ndarray:
    """Linear sRGB from chromaticities inside the locus at Y in (0, 2], what the sim produces."""
    import colour

    from spektrafilm.utils import gamut_compression as gc

    rng = np.random.default_rng(0)
    locus = gc.spectral_locus_xy()
    from matplotlib.path import Path as MplPath

    path = MplPath(locus)
    kept: list[np.ndarray] = []
    while sum(len(k) for k in kept) < 2048:
        candidates = rng.uniform([0.0, 0.0], [0.75, 0.85], size=(8192, 2))
        inside = path.contains_points(candidates)
        kept.append(candidates[inside])
    xy = np.concatenate(kept, axis=0)[:2048]
    Y = rng.uniform(1e-4, 2.0, size=(2048, 1))
    xyz = gc._xy_to_xyz_unit_y(xy) * Y
    white = np.asarray(colour.RGB_COLOURSPACES["sRGB"].whitepoint, dtype=float)
    return np.asarray(
        colour.XYZ_to_RGB(xyz, colourspace="sRGB", illuminant=white, apply_cctf_encoding=False)
    )


@fixture
def gamut_knee():
    """The Reinhard knee and the one-sided lightness knee."""
    from spektrafilm.utils import gamut_compression as gc

    yield "gamut_knee_input", KNEE_SWEEP
    for label, knee in (("default", DEFAULT_KNEE), ("soft", SOFT_KNEE), ("late", LATE_KNEE)):
        t, limit, p = knee
        yield (
            f"gamut_knee_{label}",
            gc.reinhard_knee(KNEE_SWEEP, threshold=t, limit=limit, power=p),
        )

    lightness = np.array([
        -0.2, -1e-9, 0.0, 0.1, 0.35, 0.7, 0.8, 1.0, 1.3, 2.0, 10.0, 1e9,
    ])
    yield "gamut_lightness_input", lightness
    for label, white in (
        ("white1", 1.0),
        ("whitejz", 0.16717342769906365),
        ("white100", 100.0),
    ):
        yield (
            f"gamut_lightness_{label}",
            gc._compress_lightness(lightness * white, params=(0.7, 1.0, 2.2), L_white=white),
        )


@fixture
def gamut_locus():
    """The locus polygon and the two predicates the input side asks of it."""
    from matplotlib.path import Path as MplPath

    from spektrafilm.utils import gamut_compression as gc

    locus = gc.spectral_locus_xy()
    yield "gamut_spectral_locus", locus

    # Every 5° around the circle, so the ray hits every part of the polygon including the flat
    # purple line.
    degrees = np.arange(0.0, 360.0, 5.0)
    radians = np.radians(degrees)
    directions = np.stack([np.cos(radians), np.sin(radians)], axis=-1)
    for label, white in (("white_e", WHITE_E), ("d55", WHITE_D55)):
        yield (
            f"gamut_ray_distance_{label}",
            gc._ray_polygon_distance(np.asarray(white), directions, locus),
        )

    axis = np.linspace(-0.1, 1.0, 81)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([x.ravel(), y.ravel()], axis=-1)
    path = MplPath(locus)
    yield "gamut_point_in_polygon", path.contains_points(grid).astype(np.float64)
    # Points exactly on the polygon. matplotlib calls 28 of the 131 vertices and edge midpoints
    # inside; the even-odd crossing rule the port uses calls 51, and the two disagree on 49. The
    # bisection that builds the envelope never samples a point on an edge, so the tables still
    # match.
    midpoints = 0.5 * (locus[:-1] + locus[1:])
    yield (
        "gamut_point_in_polygon_boundary",
        np.concatenate([
            path.contains_points(locus).astype(np.float64),
            path.contains_points(midpoints).astype(np.float64),
        ]),
    )


@fixture
def gamut_compress_xy():
    """Both input algorithms over the case list and a square of chromaticities."""
    from spektrafilm.utils import gamut_compression as gc
    from spektrafilm.utils.gamut_compression import InputGamutCompressSpec

    xy = xy_input()
    yield "gamut_xy_input", xy
    for algorithm in ("xy", "oklch"):
        for label, white in (("white_e", WHITE_E), ("d55", WHITE_D55)):
            spec = InputGamutCompressSpec(algorithm=algorithm, knee=DEFAULT_KNEE)
            yield (
                f"gamut_xy_{algorithm}_{label}",
                gc.compress_xy(xy, np.asarray(white), spec),
            )
    # A non-default knee, so a port that hardcodes the default knee fails here.
    spec = InputGamutCompressSpec(algorithm="xy", knee=SOFT_KNEE)
    yield "gamut_xy_xy_white_e_soft", gc.compress_xy(xy, np.asarray(WHITE_E), spec)
    # Inactive is exact identity.
    spec = InputGamutCompressSpec(active=False)
    yield "gamut_xy_inactive", gc.compress_xy(xy, np.asarray(WHITE_E), spec)

    yield "gamut_nonfinite_xy_input", NONFINITE_XY
    for algorithm in ("xy", "oklch"):
        spec = InputGamutCompressSpec(algorithm=algorithm, knee=DEFAULT_KNEE)
        with np.errstate(all="ignore"):
            yield (
                f"gamut_nonfinite_xy_{algorithm}",
                gc.compress_xy(NONFINITE_XY, np.asarray(WHITE_E), spec),
            )


@fixture
def gamut_perceptual():
    """Forward and inverse of the four perceptual transforms."""
    import colour

    from spektrafilm.utils import gamut_compression as gc

    xyz = xyz_input()
    yield "gamut_xyz_input", xyz

    lab = np.asarray(colour.XYZ_to_Oklab(xyz))
    yield "gamut_oklab_forward", lab
    yield "gamut_oklab_inverse", np.asarray(colour.Oklab_to_XYZ(lab))

    L = np.array([-0.1, 0.0, 1e-9, 0.02, 0.25, 0.5, 0.75, 1.0, 1.2, 2.0])
    yield "gamut_oklrab_l_input", L
    Lr = gc._oklab_L_to_oklrab_Lr(L)
    yield "gamut_oklrab_lr", Lr
    yield "gamut_oklrab_l_from_lr", gc._oklrab_Lr_to_oklab_L(Lr)

    jab = np.asarray(colour.XYZ_to_Jzazbz(xyz * 100.0))
    yield "gamut_jzazbz_forward", jab
    yield "gamut_jzazbz_inverse", np.asarray(colour.Jzazbz_to_XYZ(jab))

    xyz_w = gc._output_cs_whitepoint_xyz("sRGB")
    ucs = np.asarray(
        colour.XYZ_to_CAM16UCS(xyz, XYZ_w=xyz_w, L_A=64.0, Y_b=20.0)
    )
    yield "gamut_cam16ucs_forward", ucs
    yield (
        "gamut_cam16ucs_inverse",
        np.asarray(colour.CAM16UCS_to_XYZ(ucs, XYZ_w=xyz_w, L_A=64.0, Y_b=20.0)),
    )
    # The hoisted viewing-condition scalars, in the order the Swift struct stores them.
    yield "gamut_cam16_viewing_conditions", _cam16_scalars(xyz_w)


def _cam16_scalars(xyz_w: np.ndarray) -> np.ndarray:
    """D_RGB, n, F_L, N_bb, z, A_w and the chroma exponent term, for the D65 whitepoint."""
    import colour

    L_A, Y_b = 64.0, 20.0
    xyz_w100 = xyz_w * 100.0
    m16 = colour.appearance.cam16.MATRIX_16
    rgb_w = m16 @ xyz_w100
    D = np.clip(1.0 * (1.0 - (1.0 / 3.6) * np.exp((-L_A - 42.0) / 92.0)), 0, 1)
    Y_w = xyz_w100[1]
    n = Y_b / Y_w
    k = 1.0 / (5.0 * L_A + 1.0)
    k4 = k**4
    F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * (1.0 - k4) ** 2 * np.sign(5.0 * L_A) * np.abs(5.0 * L_A) ** (1.0 / 3.0)
    N_bb = 0.725 * (1.0 / n) ** 0.2
    z = 1.48 + np.sqrt(n)
    D_RGB = D * Y_w / rgb_w + 1.0 - D
    from colour.appearance.ciecam02 import post_adaptation_non_linear_response_compression_forward

    rgb_aw = post_adaptation_non_linear_response_compression_forward(D_RGB * rgb_w, F_L)
    A_w = (2.0 * rgb_aw[0] + rgb_aw[1] + rgb_aw[2] / 20.0 - 0.305) * N_bb
    return np.concatenate([
        D_RGB,
        [n, F_L, N_bb, z, A_w, (1.64 - 0.29**n) ** 0.73],
    ])


@fixture
def gamut_compress_rgb():
    """All five output algorithms, three output colour spaces, with and without the lightness knee."""
    from spektrafilm.utils import gamut_compression as gc
    from spektrafilm.utils.gamut_compression import OutputGamutCompressSpec

    rgb = rgb_input()
    yield "gamut_rgb_input", rgb

    yield "gamut_rgb_off", gc.compress_rgb(rgb, OutputGamutCompressSpec(algorithm="off"))
    yield (
        "gamut_rgb_aces_rgc",
        gc.compress_rgb(rgb, OutputGamutCompressSpec(algorithm="aces_rgc", knee=DEFAULT_KNEE)),
    )
    yield (
        "gamut_rgb_aces_rgc_soft",
        gc.compress_rgb(rgb, OutputGamutCompressSpec(algorithm="aces_rgc", knee=SOFT_KNEE)),
    )

    for algorithm in PERCEPTUAL:
        spec = OutputGamutCompressSpec(algorithm=algorithm)
        yield (
            f"gamut_rgb_{algorithm}_srgb",
            gc.compress_rgb(rgb, spec, output_color_space="sRGB"),
        )
        # Lightness compression off, on the knees the reference's own tests use.
        knee = SOFT_KNEE if algorithm in ("oklch", "jzazbz") else LATE_KNEE
        spec = OutputGamutCompressSpec(
            algorithm=algorithm, knee=knee, lightness_compression=None,
        )
        yield (
            f"gamut_rgb_{algorithm}_srgb_nolightness",
            gc.compress_rgb(rgb, spec, output_color_space="sRGB"),
        )

    for space, slug in (("Display P3", "display_p3"), ("ITU-R BT.2020", "bt2020")):
        yield (
            f"gamut_rgb_cam16ucs_{slug}",
            gc.compress_rgb(rgb, OutputGamutCompressSpec(), output_color_space=space),
        )

    # Physically realizable pixels: chromaticities inside the locus at Y in (0, 2]. This is the only
    # distribution representative of what the shipping stage does.
    realizable = realizable_input()
    yield "gamut_realizable_input", realizable
    yield (
        "gamut_realizable_cam16ucs",
        gc.compress_rgb(realizable, OutputGamutCompressSpec(), output_color_space="sRGB"),
    )

    # Negative luminance, which CAM16 turns into NaN chroma. Unreachable from the pipeline, but
    # pinned so the port's behaviour on it is recorded.
    negative = np.array([
        [-0.5, 0.1, 0.1],
        [-0.5, -0.5, -0.5],
        [-1.0, 0.0, 0.0],
        [0.0, -0.2, 0.0],
    ])
    yield "gamut_negative_input", negative
    yield (
        "gamut_negative_cam16ucs",
        gc.compress_rgb(negative, OutputGamutCompressSpec(), output_color_space="sRGB"),
    )

    yield "gamut_nonfinite_input", NONFINITE_RGB
    for algorithm in ("aces_rgc",) + PERCEPTUAL:
        space = None if algorithm == "aces_rgc" else "sRGB"
        with np.errstate(all="ignore"):
            yield (
                f"gamut_nonfinite_{algorithm}",
                gc.compress_rgb(
                    NONFINITE_RGB,
                    OutputGamutCompressSpec(algorithm=algorithm),
                    output_color_space=space,
                ),
            )


@fixture
def gamut_cmax():
    """A slice of the default path's chroma envelope, and spot rows of the others."""
    from spektrafilm.utils import gamut_compression as gc

    table = gc._get_output_c_max_table("cam16ucs", "sRGB")[2]
    # Every fourth hue. The sidecar's hashes gate the full table bit for bit; this slice makes a
    # mismatch diagnosable.
    yield "gamut_cmax_cam16ucs_srgb_h4", table[:, ::4]

    for space in ("oklch", "oklrab", "jzazbz"):
        rows = gc._get_output_c_max_table(space, "sRGB")[2][[0, 16, 32, 48, 63], :]
        yield f"gamut_cmax_{space}_srgb_rows", rows

    locus_table = gc._get_oklch_c_max_table(gc.spectral_locus_xy())[2]
    yield "gamut_cmax_locus_rows", locus_table[[0, 16, 32, 48, 63], :]

    # The bilinear lookup, including the wrap at +pi and the clamp above the grid.
    L = np.array([0.5, 0.75, 0.9, 0.02, 1.05, 0.6, 0.02, 1.0, -1.0])
    h = np.array([0.0, 2.0, -1.5, 0.0, 3.1, np.pi, -np.pi, np.pi - 1e-12, 0.5])
    yield "gamut_cmax_lookup_l", L
    yield "gamut_cmax_lookup_h", h
    yield (
        "gamut_cmax_lookup_oklch_srgb",
        gc._c_max_lookup(L, h, *gc._get_output_c_max_table("oklch", "sRGB")),
    )


@sidecar
def gamut_cmax_tables() -> str:
    """SHA-256 and statistics of every C_max envelope, as a bit-exactness gate.

    The tables are 46 080 float64 each, so committing all seven would add 2.5 MB of goldens. The
    bisection is deterministic given the same in-gamut predicate, so a hash is a cheaper and
    stricter gate than an elementwise comparison under the 1e-4 tolerance.
    """
    from spektrafilm.utils import gamut_compression as gc

    entries = {}

    def record(key: str, table: np.ndarray) -> None:
        flat = np.ascontiguousarray(table, dtype="<f8")
        entries[key] = {
            "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
            "shape": list(table.shape),
            "max": float(table.max()),
            "zero_cells": int((table == 0).sum()),
            "spot": {
                "0,0": float(table[0, 0]),
                "32,360": float(table[32, 360]),
                "63,719": float(table[63, 719]),
            },
        }

    for space in PERCEPTUAL:
        record(f"{space}|sRGB", gc._get_output_c_max_table(space, "sRGB")[2])
    for name in ("Display P3", "ITU-R BT.2020"):
        record(f"cam16ucs|{name}", gc._get_output_c_max_table("cam16ucs", name)[2])
    record("locus|oklch", gc._get_oklch_c_max_table(gc.spectral_locus_xy())[2])

    path = GOLDENS / "gamut_cmax_tables.json"
    path.write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
    return f"{len(entries)} envelope hashes -> gamut_cmax_tables.json"
