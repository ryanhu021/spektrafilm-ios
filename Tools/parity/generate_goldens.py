#!/usr/bin/env python3
"""Generate the parity goldens the Swift tests check against.

Every fixture comes from the pinned oracle (see upstream_pin.json) and is committed, so CI needs no
Python. Only re-run this after `make oracle`. A changed fixture means the render changed, so review
the delta before committing it.

Fixtures live one module per subsystem under `fixtures/`. Add one by writing a `@fixture`-decorated
generator of `(name, array)` pairs in the matching module, or in a new module. Keep inputs small and
deterministic, since every array becomes a file in git.

Usage:
    Tools/parity/oracle/.venv/bin/python Tools/parity/generate_goldens.py [fixture ...]

With no arguments it regenerates everything. Arguments are fixture-function names, and the manifest
is merged rather than replaced, so a partial run does not drop the other entries.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle_env  # noqa: E402
import spkg  # noqa: E402
from fixture_registry import registered, registered_sidecars  # noqa: E402

REPO = HERE.parents[1]
GOLDENS = REPO / "Tests" / "SpektraFilmTests" / "Goldens"


def load_fixture_modules() -> list[str]:
    """Imports every module under `fixtures/`, which registers their producers."""
    names = sorted(p.stem for p in (HERE / "fixtures").glob("*.py") if p.stem != "__init__")
    for name in names:
        importlib.import_module(f"fixtures.{name}")
    return names


def main() -> int:
    oracle_env.require()
    modules = load_fixture_modules()
    wanted = set(sys.argv[1:])

    producers = registered()
    if wanted:
        known = {fn.__name__ for fn in producers} | {
            fn.__name__ for fn in registered_sidecars()
        }
        unknown = wanted - known
        if unknown:
            print(
                f"unknown fixture(s): {', '.join(sorted(unknown))}\n"
                f"available: {', '.join(sorted(known))}",
                file=sys.stderr,
            )
            return 1
        producers = [fn for fn in producers if fn.__name__ in wanted]

    print(f"{len(modules)} fixture module(s): {', '.join(modules)}")

    manifest: dict[str, dict] = {}
    total = 0
    for fn in producers:
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

    for produce in registered_sidecars():
        if wanted and produce.__name__ not in wanted:
            continue
        print(f"  {produce.__name__:48s} {produce()}")

    manifest_path = GOLDENS / "manifest.json"
    if wanted and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.update(manifest)
        manifest = existing
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    print(f"\n{len(manifest)} fixtures in the manifest, {total:,} B written this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
