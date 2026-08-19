"""Mappers for iterators.

Mappers execute a function over the units produced by an iterator and collect
the results. A mapper is the *only* way to schedule iterator work: `map` and
`reduce` take a `mapper` argument and nothing else, so the pool size lives on
the mapper that owns it rather than being spelled a second time at the call.

Three ship with ngio. `BasicMapper` is serial and is what `mapper=None` means.
`ThreadedMapper` fans units out on a thread pool — the fit for round-trip-bound
IO and for `func`s that release the GIL. `ProcessMapper` fans out on a
spawn-based process pool — the fit for pure-Python, GIL-holding `func`s. Both
parallel mappers take their pool size at construction, and both plan their
units into conflict-free *waves* (`plan_waves`): two units whose write
footprints share a write unit never run in the same wave, and the waves run
back to back, each at full pool width. Within a wave, disjointness is what
makes the parallel writes safe *without any lock, in any topology* — a chunk
or shard object with a single writer cannot lose an update, whether the
writers are threads or processes, where a lock could only ever protect one
process. Tiling with `by_write_units()` yields a single wave. One caveat:
writes overlapping at the *pixel* level always land in different waves, and
the wave order may differ from a serial run's unit order — mask-merged and
pixel-disjoint writes are order-independent, so only genuinely overlapping
plain writes can differ from serial.
"""

import logging
import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Protocol, TypeVar, cast

from ngio.common import Roi
from ngio.common._concurrency import MaxWorkers, _resolve_max_workers
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._ops_slices_utils import (
    ChunkRect,
    chunk_rects_intersect,
    compute_chunk_rect,
)
from ngio.utils import NgioStore, NgioValueError

logger = logging.getLogger(f"ngio:{__name__}")

T = TypeVar("T")
R = TypeVar("R")


def _validate_max_workers(max_workers: MaxWorkers) -> None:
    """Refuse a pool size that cannot mean anything.

    `0` or a negative would otherwise be clamped to 1 and run serially — a
    misconfiguration silently absorbed by the very flag that asked for
    parallelism.
    """
    if isinstance(max_workers, int) and max_workers < 1:
        raise NgioValueError(
            f"max_workers must be >= 1, got {max_workers}. Use 1 for serial "
            'execution, or "auto" to size the pool automatically.'
        )


def compute_write_footprint(setter: DataSetterProtocol[Any]) -> ChunkRect | None:
    """The chunk rectangle a setter will write, at write granularity.

    The granularity is the shard shape when the target array is sharded (writes
    are atomic per shard object), the chunk shape otherwise — see
    `AbstractImage.write_granularity`. Returns `None` when the setter's
    selection is empty.
    """
    granularity = setter.zarr_array.shards or setter.slicing_ops.on_disk_chunks
    return compute_chunk_rect(
        shape=setter.slicing_ops.on_disk_shape,
        chunks=granularity,
        slicing_tuple=setter.slicing_ops.normalized_slicing_tuple,
    )


@dataclass(frozen=True, kw_only=True)
class IterUnit(Generic[T]):
    """One schedulable unit of iterator work: read one ROI, optionally write back.

    Attributes:
        index: Position in `iterator.rois` — mapper results must be returned
            in this order.
        roi: The ROI this unit covers.
        getter: Reads the ROI's data.
        setter: Writes the transformed data back, or `None` for a read-only
            unit (a read-only iterator, or any `reduce_as_*` call).
    """

    index: int
    roi: Roi
    getter: DataGetterProtocol[T]
    setter: DataSetterProtocol[T] | None

    @property
    def write_footprint(self) -> ChunkRect | None:
        """The chunk rectangle this unit writes, at write granularity.

        The granularity is the shard shape when the output array is sharded,
        the chunk shape otherwise. `None` when the unit is read-only or its
        write selection is empty — either way there is nothing to claim and
        the unit conflicts with nothing.
        """
        if self.setter is None:
            return None
        return compute_write_footprint(self.setter)


class MapperProtocol(Protocol[T, R]):
    """Protocol for mappers.

    Implementations must honour this contract:

    - The mapper *schedules* each unit's write; it is not required to invoke
      `unit.setter` eagerly or on any particular thread, only to guarantee
      that all writes are complete when `__call__` returns.
    - A unit whose `setter` is `None` is read-only: compute and collect its
      result, never write.
    - For a unit with a setter the collected result may be `None` instead of
      the written patch: `map` discards the results, and holding every patch
      until the call returns would put the whole output in memory at once.
      ngio's mappers all collect `None` for written units.
    - The returned list is ordered by `unit.index` (`results[i]` corresponds
      to `iterator.rois[i]`), regardless of execution order.
    - `units` is typically a generator and each unit is expensive to build
      (store metadata round-trips); generators are not thread-safe. A parallel
      mapper must materialise and consume `units` on the dispatching thread
      only, then distribute the materialised units to its workers.

    Compatibility policy: future ngio versions will only ever add
    *optional* members to this protocol, probed with `getattr` — an
    implementation satisfying today's contract keeps working.
    """

    def __call__(
        self,
        func: Callable[[T], R],
        units: Iterable[IterUnit[T]],
    ) -> list[R]:
        """Apply `func` to every unit and return the results in ROI order."""
        ...


class BasicMapper(Generic[T, R]):
    """Serial mapper: read, apply, write (if writable), one unit at a time."""

    def __call__(
        self,
        func: Callable[[T], R],
        units: Iterable[IterUnit[T]],
    ) -> list[R]:
        """Apply `func` to every unit and return the results in ROI order."""
        results: list[tuple[int, R]] = []
        for unit in units:
            result = func(unit.getter())
            if unit.setter is not None:
                # map_as_* is the only writing entry point and its func is
                # Callable[[T], T], so T == R whenever setter is not None;
                # reduce_as_* always builds units with setter=None. The written
                # patch is not kept: map discards the results, and holding
                # every patch until the loop ends is the whole output at once.
                unit.setter(cast("T", result))
                result = cast("R", None)
            results.append((unit.index, result))
        results.sort(key=lambda item: item[0])
        return [result for _, result in results]


def _collect_write_footprints(
    units: Sequence[IterUnit[Any]],
) -> dict[int, ChunkRect]:
    """Write footprints by `unit.index`, skipping read-only / empty writes.

    The single source of footprints for both `plan_waves` and
    `write_conflict_components` — sharing it is what guarantees the wave
    scheduler and the job splitter can never disagree about who conflicts.
    """
    footprints: dict[int, ChunkRect] = {}
    for unit in units:
        footprint = unit.write_footprint
        if footprint is not None:
            footprints[unit.index] = footprint
    return footprints


def _write_conflict_edges(
    units: Sequence[IterUnit[Any]],
    footprints: dict[int, ChunkRect],
) -> dict[int, set[int]]:
    """The adjacency of units whose write footprints share a write unit.

    Keyed and valued by `unit.index`; only units present in `footprints` (a
    setter with a non-empty selection) appear. Units targeting different
    output arrays never conflict.
    """
    adjacency: dict[int, set[int]] = {index: set() for index in footprints}
    by_array: dict[int, list[IterUnit[Any]]] = {}
    for unit in units:
        if unit.setter is None or unit.index not in footprints:
            continue
        by_array.setdefault(id(unit.setter.zarr_array), []).append(unit)

    for group in by_array.values():
        if len(group) < 2:
            continue
        # Sweep the axis with the most distinct starts (the same idea as
        # `check_if_regions_overlap`): sorted by first chunk index there, a
        # unit is compared only against those whose range is still open.
        rank = len(footprints[group[0].index])
        axis = max(
            range(rank),
            key=lambda ax: len({footprints[u.index][ax][0] for u in group}),
        )
        ordered = sorted(group, key=lambda u: footprints[u.index][axis][0])
        active: list[IterUnit[Any]] = []
        for unit in ordered:
            rect = footprints[unit.index]
            first = rect[axis][0]
            active = [u for u in active if footprints[u.index][axis][1] >= first]
            for other in active:
                if chunk_rects_intersect(footprints[other.index], rect):
                    adjacency[unit.index].add(other.index)
                    adjacency[other.index].add(unit.index)
            active.append(unit)
    return adjacency


def plan_waves(units: Sequence[IterUnit[T]]) -> list[list[IterUnit[T]]]:
    """Partition units into conflict-free waves by first-fit greedy coloring.

    Two units whose write footprints share a write unit (a chunk, or a shard
    when the output is sharded) of the same output array never share a wave:
    a wave's writes are pairwise disjoint at write granularity, so running
    one wave at a time preserves the single-writer-per-write-unit invariant
    that makes parallel writes lock-free in any topology. Read-only units and
    units with an empty write selection conflict with nothing and land in the
    first wave. Coloring runs in ascending `unit.index`, so the schedule is a
    pure function of the unit sequence. A conflict-free set — a
    `by_write_units()` tiling, say — yields a single wave.

    Note that units whose writes overlap at the *pixel* level always land in
    different waves, but the wave order may differ from a serial run's unit
    order; mask-merged and pixel-disjoint writes are order-independent either
    way.
    """
    if not units:
        return []
    adjacency = _write_conflict_edges(units, _collect_write_footprints(units))

    wave_of: dict[int, int] = {}
    waves: list[list[IterUnit[T]]] = [[]]
    for unit in sorted(units, key=lambda u: u.index):
        taken = {
            wave_of[other]
            for other in adjacency.get(unit.index, ())
            if other in wave_of
        }
        color = 0
        while color in taken:
            color += 1
        wave_of[unit.index] = color
        if color == len(waves):
            waves.append([])
        waves[color].append(unit)

    if len(waves) == len(units) and len(units) > 1:
        logger.warning(
            "Parallel map degraded to a serial schedule: every one of the "
            f"{len(units)} units writes into the same write unit(s) as another, "
            "so no two can run concurrently. Re-tile with `by_write_units()` "
            "for a single fully-parallel wave."
        )
    elif len(waves) > 1:
        largest = max(len(wave) for wave in waves)
        logger.info(
            f"Parallel map scheduled {len(units)} units into {len(waves)} "
            f"conflict-free waves (largest wave: {largest} units). Tiling with "
            "`by_write_units()` would yield a single wave."
        )
    return waves


def write_conflict_components(units: Sequence[IterUnit[Any]]) -> list[list[int]]:
    """Connected components of the write-conflict graph, as unit-index lists.

    Two units are connected when their write footprints share a write unit (a
    chunk, or a shard when the output is sharded) of the same output array —
    the same adjacency `plan_waves` schedules by, computed by the same code,
    so the scheduler and the splitter can never disagree about who conflicts.
    Read-only units and units with an empty write selection conflict with
    nothing and are singletons.

    A component is the unit of independence: units in different components
    never share a write unit, so they need no coordination in any topology.
    That is the property `split(n_jobs)` builds on — a component never spans
    two jobs. Each component is a sorted list of `unit.index`; components are
    ordered by their smallest index, and the whole result is a pure function
    of the unit sequence.
    """
    adjacency = _write_conflict_edges(units, _collect_write_footprints(units))

    components: list[list[int]] = []
    visited: set[int] = set()
    for index in sorted(unit.index for unit in units):
        if index in visited:
            continue
        visited.add(index)
        component = []
        stack = [index]
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


class ThreadedMapper(Generic[T, R]):
    """Run units concurrently on a thread pool.

    Each unit's read, `func` call and write all run on the pool. ngio's
    getters and setters are thread-safe (immutable per-ROI state over zarr's
    sync API); `func` must be thread-safe too — that part of the contract is
    the caller's.

    Before any fan-out the units are planned into conflict-free waves
    (`plan_waves`), run back to back on one pool; within a wave disjointness
    is what makes the parallel writes lock-free. With one unit, or a resolved
    pool of one, this is exactly `BasicMapper`.

    Args:
        max_workers: `"auto"` (the default) sizes the pool for
            round-trip-bound work; an `int` pins it. `None` is accepted and
            means `"auto"` — asking for this mapper *is* the opt-in.
    """

    def __init__(self, max_workers: MaxWorkers = "auto") -> None:
        _validate_max_workers(max_workers)
        self._max_workers = max_workers

    def _resolve(self, n_units: int) -> int:
        max_workers = "auto" if self._max_workers is None else self._max_workers
        resolved = _resolve_max_workers(max_workers)
        assert resolved is not None
        return max(1, min(resolved, n_units))

    def __call__(
        self,
        func: Callable[[T], R],
        units: Iterable[IterUnit[T]],
    ) -> list[R]:
        """Apply `func` to every unit and return the results in ROI order."""
        # Materialised here, on the dispatching thread: units generators are
        # not thread-safe and each unit is expensive to build.
        units = list(units)
        workers = self._resolve(len(units))
        if workers <= 1:
            return BasicMapper[T, R]()(func, units)
        waves = plan_waves(units)

        def run(unit: IterUnit[T]) -> tuple[int, R]:
            result = func(unit.getter())
            if unit.setter is not None:
                unit.setter(cast("T", result))
                # The written pixels are not shipped back: map discards them,
                # and keeping one patch per unit alive until the pool drains
                # would hold the whole output in memory.
                return unit.index, cast("R", None)
            return unit.index, result

        indexed: list[tuple[int, R]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for wave in waves:
                # Draining `pool.map` is the inter-wave barrier: the next
                # wave's writes are only submitted once this wave's are done.
                indexed.extend(pool.map(run, wave))
        indexed.sort(key=lambda item: item[0])
        return [result for _, result in indexed]


def _run_unit_in_process(func: Callable, unit: IterUnit) -> tuple[int, Any]:
    """Executed in the worker process; module-level so it pickles by reference."""
    result = func(unit.getter())
    if unit.setter is not None:
        unit.setter(result)
        # The written pixels stay in the child: shipping them back would
        # serialize every ROI through IPC for a result `map_as_*` discards.
        return unit.index, None
    return unit.index, result


def _require_process_safe_stores(units: Sequence[IterUnit[Any]]) -> None:
    """Refuse stores whose pickled copy would detach from the original.

    A `MemoryStore` pickles *by value*: each worker process would read from —
    and write into — its own private copy, and the parent would never see a
    byte of it. Local and remote stores pickle by reference (a path, a URL)
    and reopen in the child.
    """
    for unit in units:
        arrays = [unit.getter.zarr_array]
        if unit.setter is not None:
            arrays.append(unit.setter.zarr_array)
        for zarr_array in arrays:
            if NgioStore.ensure(zarr_array.store).store_type == "memory":
                raise NgioValueError(
                    "ProcessMapper cannot run on an in-memory store: a "
                    "MemoryStore pickles by value, so each worker process "
                    "would write into its own private copy and the results "
                    "would be silently lost. Use a local or remote store, or "
                    "ThreadedMapper."
                )


class ProcessMapper(Generic[T, R]):
    """Run units on a spawn-based process pool.

    The fit for pure-Python, GIL-holding `func`s (`ThreadedMapper` already
    covers IO-bound work and GIL-releasing compute). Each child reads its ROI
    from the store, applies `func`, and writes back — pixels never cross the
    process boundary. For written units the returned result is therefore
    `None` (which `map_as_*` discards anyway); only `reduce_as_*` results are
    pickled back to the parent.

    The same wave planning as `ThreadedMapper` applies, and it is the entire
    cross-process safety argument: within a wave the write units are
    disjoint, so they need no lock — and no lock ngio could take would work
    across processes anyway.

    Constraints:
    - `func` (and any transforms the units carry) must be picklable — a
      module-level function, not a lambda or closure.
    - In-memory stores are refused; see `_require_process_safe_stores`.
    - The pool uses the `spawn` start method: the parent holds zarr's IO
      event-loop thread, and forking a threaded process is unsafe. Each
      worker pays an interpreter start and `import ngio` (~1 s), amortized
      over the units it processes.

    Args:
        max_workers: `"auto"` (the default) or an `int`; `None` means
            `"auto"`. The pool never exceeds the unit count.
    """

    def __init__(self, max_workers: MaxWorkers = "auto") -> None:
        _validate_max_workers(max_workers)
        self._max_workers = max_workers

    def _resolve(self, n_units: int) -> int:
        max_workers = "auto" if self._max_workers is None else self._max_workers
        resolved = _resolve_max_workers(max_workers)
        assert resolved is not None
        return max(1, min(resolved, n_units))

    def __call__(
        self,
        func: Callable[[T], R],
        units: Iterable[IterUnit[T]],
    ) -> list[R]:
        """Apply `func` to every unit and return the results in ROI order."""
        units = list(units)
        workers = self._resolve(len(units))
        if workers <= 1:
            return BasicMapper[T, R]()(func, units)
        _require_process_safe_stores(units)
        waves = plan_waves(units)

        context = multiprocessing.get_context("spawn")
        indexed: list[tuple[int, Any]] = []
        # One pool for every wave: each worker pays its interpreter start and
        # `import ngio` exactly once, however many waves the plan has.
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for wave in waves:
                indexed.extend(pool.map(partial(_run_unit_in_process, func), wave))
        indexed.sort(key=lambda item: item[0])
        return [result for _, result in indexed]
