"""Case enumeration and measurement. Nothing here knows what ngio is.

A block module declares three things and no more:

    AXES: dict[str, list | dict]      what to sweep; may be empty
    REPEATS: int                      optional, default 5
    def run(root: Path, **values) -> Measured

`run` does its own setup and returns the callable to measure. Everything
outside that callable is excluded from the measurement -- the same split as
`Scenario(setup, run)` in `tests/performance/scenarios.py`, expressed as one
function because a block's setup and its measured call always share state.

An axis is declared in one of two forms, and the form is the meaning:

| declaration                      | kind   | the CLI can                    |
| -------------------------------- | ------ | ------------------------------ |
| `"z": [16, 64, 256]`             | open   | subset, and add new values     |
| `"layout": {"sharded": {...}}`   | closed | subset only                    |

Values on an open axis are scalars, so `--axis z=1024` can name one that was
never declared. Values on a closed axis are structures -- a dict of
`create_empty_ome_zarr` kwargs, say -- so its labels are the vocabulary and a
shard shape is not something anyone should type at a shell prompt.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

MB = 1024 * 1024

#: One axis override: `(block or None, axis name, value tokens)`. Produced by
#: a `--axis` flag and by a config file's `[axes]`, which lower to the same
#: triple so the two sources cannot drift.
Override = tuple[str | None, str, list[str]]


class Skip(Exception):  # noqa: N818
    """Raised by a block's `run` when a case in the product does not apply.

    The cartesian product is rectangular; some blocks are not. `algorithms`
    sweeps three kernels whose useful `n` ranges do not overlap, and an
    O(n^2) kernel at n=50_000 would not return. Raising from inside `run`
    keeps the constraint next to the comment that justifies it, which a
    product-filter hook on the runner would not.
    """


class Measured(NamedTuple):
    """What a block hands back: the callable to time, and an optional note."""

    fn: Callable[[], object]
    note: str = ""


class Case(NamedTuple):
    """One point of a block's cartesian product."""

    block: str
    labels: dict[str, str]
    values: dict[str, Any]

    @property
    def label(self) -> str:
        """`mode=numpy z=256`, or the block name when there are no axes."""
        return " ".join(f"{k}={v}" for k, v in self.labels.items()) or self.block


class Result(NamedTuple):
    """One measured case."""

    case: Case
    seconds: float
    peak_mb: float
    note: str = ""


def normalize(axis: list[Any] | Mapping[str, Any]) -> dict[str, Any]:
    """Return `{label: value}` for either declaration form."""
    return dict(axis) if isinstance(axis, dict) else {str(v): v for v in axis}


def cases(block: str, axes: Mapping[str, Any]) -> Iterator[Case]:
    """Enumerate the cartesian product of `axes`; the last varies fastest."""
    normalized = {name: normalize(axis) for name, axis in axes.items()}
    names = list(normalized)
    for point in product(*(normalized[name].items() for name in names)):
        yield Case(
            block,
            {n: label for n, (label, _) in zip(names, point, strict=True)},
            {n: value for n, (_, value) in zip(names, point, strict=True)},
        )


def _coerce(sample: Any, raw: str) -> Any:
    """Parse `raw` as the type of `sample`, falling back to the string."""
    if isinstance(sample, bool):
        return raw.lower() in ("1", "true", "yes")
    for kind in (type(sample), int, float):
        try:
            return kind(raw)
        except (TypeError, ValueError):
            continue
    return raw


def apply_overrides(
    block: str,
    axes: Mapping[str, Any],
    overrides: list[Override],
) -> dict[str, dict[str, Any]]:
    """Return `{axis: {label: value}}` with the axis overrides applied.

    Overrides come from `--axis` flags and from a config file's `[axes]`
    table, which lowers to the same triples so the two cannot drift.

    An override replaces an axis outright rather than extending it. On a
    closed axis it names labels; on an open axis it names values, re-parsed
    with the type of the first declared one so a value that was never
    declared can still be swept.

    An *unqualified* override on a block that has no such axis is skipped --
    that is what lets `--axis z=512` reach several blocks at once. A
    *qualified* one is checked strictly, because `--axis consolidate.zz=1`
    can only be a typo, and an override that silently measures the defaults
    is worse than no override at all.
    """
    resolved = {name: normalize(axis) for name, axis in axes.items()}
    for target, name, raw in overrides:
        if target not in (None, block):
            continue
        if name not in resolved:
            if target is None:
                continue
            raise SystemExit(
                f"{block} has no axis {name!r}; it has: {', '.join(resolved) or 'none'}"
            )
        declared = resolved[name]
        if isinstance(axes[name], dict):
            unknown = [token for token in raw if token not in declared]
            if unknown:
                raise SystemExit(
                    f"{block}.{name} has no {unknown[0]!r}; "
                    f"choose from: {', '.join(declared)}"
                )
            resolved[name] = {token: declared[token] for token in raw}
        else:
            sample = next(iter(declared.values()), "")
            resolved[name] = {token: _coerce(sample, token) for token in raw}
    return resolved


def measure(
    fn: Callable[[], Any], *, repeats: int = 5, warmup: int = 1, mem_repeats: int = 2
) -> tuple[float, float]:
    """Return (median seconds, peak MiB allocated) for `fn`.

    Timing and allocation are measured in two separate phases on purpose.
    `tracemalloc` hooks every allocation and inflates allocation-heavy code
    several-fold, and it does not inflate every case equally -- the mode that
    allocates most is penalised most, which is exactly the comparison the
    `consolidate` block exists to make. A single fused loop reports a time
    that is not the time.

    Peak is the max over its runs, not the last one: the question peak answers
    is "will this fit", and that is a worst case.

    `tracemalloc` counts Python-side allocations only. It misses memory that
    numpy/dask acquire outside the Python allocator, so treat it as a strong
    relative signal between cases rather than an absolute footprint.
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    tracemalloc.start()
    try:
        peaks = []
        for _ in range(mem_repeats):
            tracemalloc.reset_peak()
            fn()
            peaks.append(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()

    return statistics.median(times), max(peaks) / MB
