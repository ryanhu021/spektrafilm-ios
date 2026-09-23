"""Golden for the parameter defaults.

The defaults define the render. One transcription slip shifts every photo, and no other test would
catch it. So the whole default tree is dumped as JSON keyed by its Python path, and the Swift side
compares it field by field.

This emits JSON because the tree mixes numbers, booleans, strings and nulls.
Package.swift copies the Goldens directory wholesale, so the file reaches the test bundle without
any manifest change.
"""

from __future__ import annotations

import dataclasses as dc
import json
from pathlib import Path

from fixture_registry import sidecar

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "Tests" / "SpektraFilmTests" / "Goldens"


def _walk(obj, prefix: str, out: dict) -> None:
    from spektrafilm.profiles.io import Profile

    for field in dc.fields(obj):
        value = getattr(obj, field.name)
        path = f"{prefix}{field.name}"
        if isinstance(value, Profile):
            # The two profiles are measured data, covered by ProfileTests.
            continue
        if dc.is_dataclass(value):
            _walk(value, path + ".", out)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                out[f"{path}[{i}]"] = item
        elif value is None or isinstance(value, (bool, int, float, str)):
            out[path] = value
        else:
            out[path] = str(value)


@sidecar
def params_defaults() -> str:
    """Flat map of every default in RuntimePhotoParams, keyed by its Python path."""
    from spektrafilm.profiles.io import Profile
    from spektrafilm.runtime.params_schema import RuntimePhotoParams

    params = RuntimePhotoParams(film=Profile(), print=Profile())
    flat: dict = {}
    _walk(params, "", flat)

    path = GOLDENS / "params_defaults.json"
    path.write_text(json.dumps(flat, indent=1, sort_keys=True) + "\n")
    return f"{len(flat)} defaults -> params_defaults.json"
