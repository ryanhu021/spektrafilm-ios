"""Registry for golden fixture producers.

Fixtures live in one module per subsystem under `fixtures/`, so several people (or several agents)
can add fixtures at once without editing the same file. `generate_goldens.py` imports every module
in that directory and collects whatever registered itself here.
"""

from __future__ import annotations

from typing import Callable, Iterator

Producer = Callable[[], Iterator[tuple[str, object]]]

_FIXTURES: list[Producer] = []


def fixture(fn: Producer) -> Producer:
    """Registers a generator of `(name, array)` pairs.

    Keep inputs small and deterministic. Every array becomes a file in git.
    """
    _FIXTURES.append(fn)
    return fn


def registered() -> list[Producer]:
    return list(_FIXTURES)
