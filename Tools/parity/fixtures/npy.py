"""Goldens for the `.npy` reader: header variants, plus slices of the real spectra LUT.

Two halves.

`npy_case_files` writes the synthetic `.npy` files the header parser is tested against, and
`npy_case_values` yields what each one should decode to. Both walk the same `CASES` table, so the
files and their expected values cannot drift. Regenerate them together:

    generate_goldens.py npy_case_files npy_case_values npy_lut

`npy_lut` samples `Sources/SpektraFilm/Resources/luts/spectral_upsampling/irradiance_xy_tc.npy`,
the 192x192x81 float16 table the engine loads at runtime. Widening float16 to float64 is lossless,
so the Swift side is gated at tolerance 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.lib.format as npy_format

from fixture_registry import fixture, sidecar

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "Tests" / "SpektraFilmTests" / "Goldens"
LUT = REPO / "Sources" / "SpektraFilm" / "Resources" / "luts" / "spectral_upsampling" / "irradiance_xy_tc.npy"

# float16 corners of the representable range, so the subnormal and NaN paths of the widening are
# covered by a file rather than by a hand-written Swift constant.
F2_SPECIALS = [
    np.nan,
    np.inf,
    -np.inf,
    -0.0,
    0.0,
    5.960464477539063e-08,  # smallest subnormal
    6.103515625e-05,  # smallest normal
    65504.0,  # largest finite
    1.0009765625,  # 1 + 2^-10
]

def f2_sweep() -> np.ndarray:
    """Every 61st float16 bit pattern, plus the boundaries between the widening's branches.

    61 is coprime with 1024, so the sample walks the whole significand as it walks the exponent.
    The explicit patterns are zero, both subnormal ends, the largest finite, both infinities, and
    signalling as well as quiet NaN, which is where a hand-rolled widening goes wrong.
    """
    patterns = list(range(0, 65536, 61))
    patterns += [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400,
        0x7BFF, 0x7C00, 0x7C01, 0x7DFF, 0x7E00, 0xFC00, 0xFE00, 0xFFFF,
    ]
    return np.array(patterns, dtype=np.uint16).view(np.float16)


# (file stem, npy format version, array). Readable cases first, rejected cases last.
CASES: list[tuple[str, tuple[int, int], np.ndarray]] = [
    ("npy_case_v1_f8_2d", (1, 0), np.arange(12, dtype="<f8").reshape(3, 4) / 7.0),
    ("npy_case_v2_f8_2d", (2, 0), np.arange(12, dtype="<f8").reshape(3, 4) / 7.0),
    (
        "npy_case_v1_f4_1d",
        (1, 0),
        np.array([0.0, 1 / 3, -1 / 3, 1e-8, 1e8, 3.4028235e38, -2.5], dtype="<f4"),
    ),
    ("npy_case_v1_f2_3d", (1, 0), np.linspace(-3.0, 3.0, 24, dtype="<f8").reshape(2, 3, 4).astype("<f2")),
    ("npy_case_v2_f2_3d", (2, 0), np.linspace(-3.0, 3.0, 24, dtype="<f8").reshape(2, 3, 4).astype("<f2")),
    ("npy_case_v1_f2_specials", (1, 0), np.array(F2_SPECIALS, dtype="<f2")),
    ("npy_case_v1_f2_sweep", (1, 0), f2_sweep()),
    ("npy_case_v1_f8_scalar", (1, 0), np.array(3.5, dtype="<f8")),
    ("npy_case_v1_f8_empty", (1, 0), np.zeros((0,), dtype="<f8")),
    ("npy_case_v1_f8_empty_2d", (1, 0), np.zeros((0, 3), dtype="<f8")),
    ("npy_case_bad_fortran", (1, 0), np.asfortranarray(np.arange(12, dtype="<f8").reshape(3, 4))),
    ("npy_case_bad_bigendian", (1, 0), np.arange(6, dtype=">f8")),
    ("npy_case_bad_int", (1, 0), np.arange(6, dtype="<i4")),
]

# Cases the Swift reader must reject. No value golden for these.
REJECTED = {"npy_case_bad_fortran", "npy_case_bad_bigendian", "npy_case_bad_int"}

# Cases whose payload is empty or a single scalar: asserted inline in Swift, not against a .spkg,
# because a zero-element golden makes every comparison trivially pass.
UNGOLDENED = REJECTED | {"npy_case_v1_f8_empty", "npy_case_v1_f8_empty_2d"}


@sidecar
def npy_case_files() -> str:
    """Writes the synthetic .npy files under Goldens/. Package.swift copies that directory
    wholesale, so they reach the test bundle with no manifest change."""
    GOLDENS.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem, version, array in CASES:
        path = GOLDENS / f"{stem}.npy"
        with open(path, "wb") as handle:
            npy_format.write_array(handle, array, version=version)
        total += path.stat().st_size
    return f"{len(CASES)} synthetic .npy files, {total:,} B"


@fixture
def npy_case_values():
    """What each readable synthetic file decodes to under np.double."""
    for stem, _, array in CASES:
        if stem in UNGOLDENED:
            continue
        with np.errstate(invalid="ignore"):
            # Casting a signalling float16 NaN raises FPE_INVALID; the quieted result is the point.
            widened = np.double(array)
        # The .spkg container needs a rank, and a 0-d array has none. Ship it as (1,).
        yield f"{stem}_values", np.atleast_1d(widened)


@fixture
def npy_lut():
    """Corners, axis walks, a 3-D block and the extrema of the shipped spectra LUT."""
    raw = np.load(LUT)
    assert raw.dtype == np.float16 and raw.shape == (192, 192, 81), (raw.dtype, raw.shape)
    lut = np.double(raw)

    # (0, 0) and (0, 191) both decode to xy = (1, 0) but store different spectra, so a reader that
    # collapsed the first axis would still look plausible without them.
    yield "npy_lut_corners", lut[[0, 0, 191, 191, 96], [0, 191, 0, 191, 96], :]
    # One walk per axis catches a transposed or mis-strided read.
    yield "npy_lut_axis0_walk", lut[:, 77, 12]
    yield "npy_lut_axis1_walk", lut[40, :, 33]
    yield "npy_lut_block", lut[5:9, 10:13, :]
    # Summed left to right in Python, which is the order the Swift test sums in. NumPy's pairwise
    # sum, math.fsum and this loop all agree to the last bit on this data.
    running = 0.0
    for value in lut.ravel().tolist():
        running += value
    yield "npy_lut_extrema", np.array([lut.min(), lut.max(), running])
