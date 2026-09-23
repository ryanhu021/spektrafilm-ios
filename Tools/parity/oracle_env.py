"""Assert the oracle environment matches upstream_pin.json before generating anything.

The goldens and the generated tables depend on exact library versions, not just on upstream's
commit:

- colour-science supplies the colourimetric tables and the transfer functions.
- NumPy's C implementation of `np.interp` supplies the guess-threaded binary search that decides
  the DIR-coupler result for positive stocks.
- SciPy's Akima interpolator is baked into the committed filter tables.
- Numba's `fastmath` governs what `fast_interp` does with NaN.

A patch release of any of these can move a fixture, which would look like a render regression with
no code change behind it. Both generators call `require()` first so a mismatched environment fails
before it overwrites a committed fixture.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PIN = Path(__file__).resolve().parent / "upstream_pin.json"


def expected() -> dict[str, str]:
    return json.loads(PIN.read_text())["oracle_environment"]["packages"]


def require(*, strict: bool = True) -> None:
    """Raises SystemExit when an installed version differs from the pin.

    Pass `strict=False` to warn instead, for testing a new version before re-pinning.
    """
    problems = []
    for package, want in expected().items():
        try:
            have = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed, pin wants {want}")
            continue
        if have != want:
            problems.append(f"{package} is {have}, pin wants {want}")

    if not problems:
        return

    header = "oracle environment does not match Tools/parity/upstream_pin.json:"
    body = "\n".join(f"  {p}" for p in problems)
    hint = (
        "\nRun `make oracle` to rebuild it. If the new versions are intended, update the pin, "
        "regenerate every golden, and review the render delta."
    )
    if strict:
        raise SystemExit(f"{header}\n{body}{hint}")
    print(f"warning: {header}\n{body}{hint}", file=sys.stderr)
