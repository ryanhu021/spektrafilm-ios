"""Registry for golden fixture producers.

Fixtures live in one module per subsystem under `fixtures/`, so several people can add fixtures at
once without editing the same file. `generate_goldens.py` imports every module in that directory and
collects whatever registered itself here.

Two kinds:

- `@fixture` produces `(name, array)` pairs, written as .spkg and listed in the manifest. Almost
  every fixture uses this.
- `@sidecar` writes its own file under Goldens/ and returns a short description. It is for goldens
  that are not float arrays, such as the parameter defaults, which mix numbers, booleans, strings
  and nulls.
"""

from __future__ import annotations

from typing import Callable, Iterator

Producer = Callable[[], Iterator[tuple[str, object]]]
Sidecar = Callable[[], str]

_FIXTURES: list[Producer] = []
_SIDECARS: list[Sidecar] = []


def fixture(fn: Producer) -> Producer:
    """Registers a generator of `(name, array)` pairs.

    Keep inputs small and deterministic. Every array becomes a file in git.
    """
    _FIXTURES.append(fn)
    return fn


def sidecar(fn: Sidecar) -> Sidecar:
    """Registers a producer that writes its own file and returns a one-line description."""
    _SIDECARS.append(fn)
    return fn


def registered() -> list[Producer]:
    return list(_FIXTURES)


def registered_sidecars() -> list[Sidecar]:
    return list(_SIDECARS)
