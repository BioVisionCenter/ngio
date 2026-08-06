"""Benchmark declaration and registry.

A benchmark is declared once and consumed twice: by the op-count gate under
`tests/benchmarks/` and by the wall-clock layer under `benchmarks/timing/`.
Keeping a single declaration is what stops the two layers drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BenchContext", "Benchmark", "benchmark", "registry"]

Area = Literal["images", "hcs", "iterators", "tables"]
Tier = Literal["ci", "large"]


class BenchContext(Protocol):
    """What a benchmark's `setup`/`run` receives to reach its fixture."""

    def store_for(self, fixture: str) -> Any:
        """Return a counting store rooted at the named fixture."""
        ...


@dataclass(frozen=True)
class Benchmark:
    """A single measured scenario.

    Attributes:
        setup: Runs before measurement and is excluded from it. Use it for the
            part of a scenario that is not under test, so that "open once, then
            read 16 ROIs" attributes only the reads.
        requires: Store capabilities the scenario needs. Profiles that do not
            declare them are skipped rather than failed.
    """

    name: str
    area: Area
    fixture: str
    run: Callable[[Any], object]
    setup: Callable[[BenchContext], Any] | None = None
    requires: frozenset[str] = field(default_factory=frozenset)
    tier: Tier = "ci"


_REGISTRY: dict[str, Benchmark] = {}


def benchmark(
    *,
    name: str,
    area: Area,
    fixture: str,
    setup: Callable[[BenchContext], Any] | None = None,
    requires: frozenset[str] | set[str] | tuple[str, ...] = (),
    tier: Tier = "ci",
) -> Callable[[Callable[[Any], object]], Callable[[Any], object]]:
    """Register the decorated callable as a benchmark.

    The function is returned unchanged, so a scenario module stays importable
    and directly callable outside the harness.

    Raises:
        ValueError: If `name` is already registered.
    """

    def decorate(fn: Callable[[Any], object]) -> Callable[[Any], object]:
        if name in _REGISTRY:
            raise ValueError(f"duplicate benchmark name: {name!r}")
        _REGISTRY[name] = Benchmark(
            name=name,
            area=area,
            fixture=fixture,
            run=fn,
            setup=setup,
            requires=frozenset(requires),
            tier=tier,
        )
        return fn

    return decorate


def registry(*, tier: Tier | None = "ci", area: Area | None = None) -> list[Benchmark]:
    """Return registered benchmarks, sorted by name.

    Args:
        tier: Keep only this tier, or every tier when `None`.
        area: Keep only this area, or every area when `None`.
    """
    import benchmarks.scenarios  # noqa: F401  (registration side effect)

    items = _REGISTRY.values()
    if tier is not None:
        items = [b for b in items if b.tier == tier]
    if area is not None:
        items = [b for b in items if b.area == area]
    return sorted(items, key=lambda b: b.name)
