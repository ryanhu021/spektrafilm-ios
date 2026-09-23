"""Minimal container for parity goldens.

One fixture per file, so a Swift test can read a NumPy array with no dependency and a truncated
file fails loudly instead of reading as zeros. `.npy` would also work, but its header carries dtype
strings, Fortran order and shape tuples to parse.

Layout, little-endian throughout:

    0   8 bytes   magic "SPKG0001"
    8   4 bytes   uint32 rank
    12  4*rank    uint32 dimensions, C order
    ..  pad to a multiple of 8
    ..  8*count   float64 values, C order

float64 only. The reference computes in float64, and a fixture silently narrowed to float32 would
shift the goldens under the 1e-4 gate.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"SPKG0001"


def write(path: str | Path, array: np.ndarray) -> int:
    """Writes `array` as float64 and returns the byte count."""
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    header = MAGIC + struct.pack("<I", a.ndim) + b"".join(
        struct.pack("<I", d) for d in a.shape
    )
    header += b"\0" * (-len(header) % 8)
    payload = header + a.tobytes(order="C")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


def read(path: str | Path) -> np.ndarray:
    """Reads a fixture back, for round-trip checks."""
    blob = Path(path).read_bytes()
    if blob[:8] != MAGIC:
        raise ValueError(f"{path}: not a spkg file")
    (rank,) = struct.unpack_from("<I", blob, 8)
    shape = tuple(struct.unpack_from("<I", blob, 12 + 4 * i)[0] for i in range(rank))
    offset = 12 + 4 * rank
    offset += -offset % 8
    count = int(np.prod(shape)) if shape else 1
    values = np.frombuffer(blob, dtype="<f8", count=count, offset=offset)
    return values.reshape(shape)
