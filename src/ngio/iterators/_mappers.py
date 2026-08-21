"""Mappers for iterators.

A mapper executes a function over the units an iterator produces and
collects the results; `map` and `reduce` take a `mapper` argument and
nothing else, so the pool size lives on the mapper that owns it.

Four ship with ngio: `BasicMapper` (serial, what `mapper=None` means),
`BatchedMapper` (serial, one stacked `(B, ...)` array per `func` call),
`ThreadedMapper` (thread pool), and `ProcessMapper` (spawned processes).
Every mapper runs the units in the same canonical order — flattened
conflict-free waves; `plan_waves` carries the safety argument — so
overlapping writes land identically whichever mapper runs them: the later
wave wins, deterministically. With no conflicts the canonical order is
plain index order, and `by_write_units()` yields a single fully-parallel
wave.
"""

import logging
import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Protocol, TypeVar, cast

import numpy as np

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


def _is_same_zarr_array(left, right) -> bool:
    """Whether two zarr array handles point at the same stored array.

    Unknowable — a store whose comparison raises — counts as *same*: every
    caller uses "same" as the conservative answer. `with_halo` refuses
    in-place iteration, the conflict graph adds an edge (over-serialising
    instead of allowing a lost update).
    """
    if left is right:
        return True
    try:
        return bool(left.store == right.store and left.path == right.path)
    except (AttributeError, TypeError):
        return True


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
            unit (a read-only iterator, or any `reduce` call).
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
        """Apply `func` to every unit and return the results in ROI order.

        Units run in canonical (flattened wave) order, so pixel-overlapping
        writes land exactly as they would under a parallel mapper.
        """
        results: list[tuple[int, R]] = []
        for unit in canonical_unit_order(list(units)):
            result = func(unit.getter())
            if unit.setter is not None:
                # `map` is the only writing entry point and its func is
                # Callable[[T], T], so T == R whenever setter is not None;
                # `reduce` always builds units with setter=None. The written
                # patch is not kept: map discards the results, and holding
                # every patch until the loop ends is the whole output at once.
                unit.setter(cast("T", result))
                result = cast("R", None)
            results.append((unit.index, result))
        results.sort(key=lambda item: item[0])
        return [result for _, result in results]


class BatchedMapper:
    """Stack patches into `(B, ...)` batches and call `func` once per batch.

    The fit for neural-network inference. Unlike every other mapper, `func`
    receives a *stacked* array — a leading batch axis over up to
    `batch_size` patches — and must return an array-like with the same
    leading axis whose items follow `map`'s per-patch contract.

    Ragged tilings are padded per batch to the per-axis maximum before
    stacking (origin-anchored: real pixels first, padding after); a
    shape-preserving output is sliced back to each patch's true shape
    before the write, and halos are trimmed by the setter as usual. A
    per-item reduction is allowed only on a uniform batch — on a ragged
    one its result is computed on padded pixels and there is no padding
    to slice back off, so it raises. Reads within a batch fan out on a
    thread pool; writes run serially on the dispatching thread, so
    batched mapping is write-safe on any tiling.

    Stacking is a numpy operation, so unlike its siblings this mapper is
    not generic over the payload type: it accepts bare `np.ndarray` units
    only (tuple payloads raise). Its pool argument is named
    `read_workers`, not `max_workers`, because it sizes something
    different — the other mappers' pools run `func`, this one's only
    reads; `func` always runs once per batch on the dispatching thread.

    Args:
        batch_size: Number of patches stacked per `func` call. The last
            batch may be smaller.
        pad_mode: `np.pad` mode used to grow ragged patches to the batch
            shape, typically `"constant"` or `"reflect"`.
        pad_values: Fill value when `pad_mode="constant"`, ignored otherwise.
        read_workers: Pool size for the per-batch reads. `"auto"` (the
            default) sizes it for round-trip-bound work, an `int` pins it,
            `1` reads serially; `None` means `"auto"`.
    """

    def __init__(
        self,
        batch_size: int = 8,
        pad_mode: str = "constant",
        pad_values: int | float = 0,
        read_workers: MaxWorkers = "auto",
    ) -> None:
        if batch_size < 1:
            raise NgioValueError(f"batch_size must be >= 1, got {batch_size}.")
        _validate_max_workers(read_workers)
        self._batch_size = batch_size
        self._pad_mode = pad_mode
        self._pad_values = pad_values
        self._read_workers = read_workers

    def _resolve_read_workers(self) -> int:
        read_workers = "auto" if self._read_workers is None else self._read_workers
        resolved = _resolve_max_workers(read_workers)
        assert resolved is not None
        return max(1, min(resolved, self._batch_size))

    def _pad(self, patch: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
        # Origin-anchored: real pixels first, padding after, so the way back
        # is a plain `[:size]` slice per axis.
        pad_widths = tuple(
            (0, target - size)
            for size, target in zip(patch.shape, batch_shape, strict=True)
        )
        if not any(after for _, after in pad_widths):
            return patch
        if self._pad_mode == "constant":
            return np.pad(
                patch, pad_widths, mode="constant", constant_values=self._pad_values
            )
        return np.pad(patch, pad_widths, mode=self._pad_mode)  # ty: ignore[no-matching-overload]

    def _process_batch(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        batch: Sequence[IterUnit[np.ndarray]],
        patches: Sequence[np.ndarray],
    ) -> list[tuple[int, np.ndarray]]:
        for unit, patch in zip(batch, patches, strict=True):
            if not isinstance(patch, np.ndarray):
                raise NgioValueError(
                    "BatchedMapper can only stack bare-array units, got "
                    f"{type(patch).__name__} for ROI {unit.roi.name!r}. Iterators "
                    "whose units carry tuples (feature extraction, object "
                    "detection) cannot be batched."
                )
        try:
            batch_shape = tuple(
                max(sizes)
                for sizes in zip(*(patch.shape for patch in patches), strict=True)
            )
        except ValueError:
            ndims = sorted({patch.ndim for patch in patches})
            raise NgioValueError(
                f"Cannot stack patches of mixed dimensionality {ndims} into one batch."
            ) from None

        stacked = np.stack([self._pad(patch, batch_shape) for patch in patches])
        out = np.asarray(func(stacked))
        if out.shape[:1] != (len(batch),):
            raise NgioValueError(
                f"The batched func returned shape {out.shape} for a batch of "
                f"{len(batch)} patches; the leading axis must be the batch axis."
            )

        # Slice each item back to its patch's pre-padding shape — but only
        # when `func` preserved the item shape. On a ragged batch any other
        # output shape is refused: it was computed on padded input, so the
        # padding has already leaked into the values, and there is no
        # padding to slice back off a shape-changing result. On a uniform
        # batch nothing was padded, so any output shape passes through (a
        # writing unit's setter still rejects a wrong shape loudly).
        trim = tuple(out.shape[1:]) == batch_shape
        ragged = any(patch.shape != batch_shape for patch in patches)
        if ragged and not trim:
            raise NgioValueError(
                f"The batched func returned items of shape "
                f"{tuple(out.shape[1:])} for a ragged batch padded to "
                f"{batch_shape}: the result was computed on padded patches, "
                "and a shape-changing result cannot have the padding sliced "
                "back off. Tile uniformly, or return one item per patch in "
                "the padded shape."
            )
        indexed: list[tuple[int, np.ndarray]] = []
        for unit, patch, item in zip(batch, patches, out, strict=True):
            if trim and patch.shape != batch_shape:
                item = item[tuple(slice(0, size) for size in patch.shape)]
            if unit.setter is not None:
                unit.setter(item)
                # As everywhere else: the written patch is not shipped back.
                indexed.append((unit.index, cast("np.ndarray", None)))
            else:
                indexed.append((unit.index, item))
        return indexed

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        units: Iterable[IterUnit[np.ndarray]],
    ) -> list[np.ndarray]:
        """Apply `func` batch-wise and return the results in ROI order.

        Batches are cut over the canonical (flattened wave) unit order, so
        pixel-overlapping writes land exactly as under every other mapper;
        with no conflicts the batches follow plain index order.
        """
        units = canonical_unit_order(list(units))
        if not units:
            return []
        batches = [
            units[start : start + self._batch_size]
            for start in range(0, len(units), self._batch_size)
        ]
        workers = min(self._resolve_read_workers(), len(units))
        indexed: list[tuple[int, np.ndarray]] = []
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for batch in batches:
                if pool is not None:
                    patches = list(pool.map(lambda unit: unit.getter(), batch))
                else:
                    patches = [unit.getter() for unit in batch]
                indexed.extend(self._process_batch(func, batch, patches))
        finally:
            if pool is not None:
                pool.shutdown()
        indexed.sort(key=lambda item: item[0])
        return [result for _, result in indexed]


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


def _collect_extra_claims(
    units: Sequence[IterUnit[Any]],
) -> dict[Any, list[tuple[int, ChunkRect]]]:
    """Side-channel write claims by claim key, as `(unit.index, rect)` pairs.

    A setter that writes anywhere besides its own target array may declare
    those writes through the optional `extra_write_footprints` member
    (probed with `getattr`, per the mapper compatibility policy): `(key,
    rect)` pairs, where the key identifies the side array and the rect is
    the claim on its chunk grid. Claims under the same key conflict like
    footprints on one array; claims under different keys never do. No ngio
    setter declares any today — the stitch banks into per-tile arrays with a
    single writer each — but the protocol stays for third-party setters
    (`HaloCroppingSetter` forwards it).
    """
    claims: dict[Any, list[tuple[int, ChunkRect]]] = {}
    for unit in units:
        if unit.setter is None:
            continue
        extra = getattr(unit.setter, "extra_write_footprints", None)
        if not extra:
            continue
        for key, rect in extra:
            claims.setdefault(key, []).append((unit.index, rect))
    return claims


def _sweep_adjacency(
    pairs: list[tuple[int, ChunkRect]],
    adjacency: dict[int, set[int]],
) -> None:
    """Add edges between intersecting rects of one array into `adjacency`.

    Sweeps the axis with the most distinct starts (the same idea as
    `check_if_regions_overlap`): sorted by first chunk index there, a rect is
    compared only against those whose range is still open.
    """
    if len(pairs) < 2:
        return
    rank = len(pairs[0][1])
    axis = max(
        range(rank),
        key=lambda ax: len({rect[ax][0] for _, rect in pairs}),
    )
    ordered = sorted(pairs, key=lambda pair: pair[1][axis][0])
    active: list[tuple[int, ChunkRect]] = []
    for index, rect in ordered:
        first = rect[axis][0]
        active = [pair for pair in active if pair[1][axis][1] >= first]
        for other_index, other_rect in active:
            if other_index != index and chunk_rects_intersect(other_rect, rect):
                adjacency.setdefault(index, set()).add(other_index)
                adjacency.setdefault(other_index, set()).add(index)
        active.append((index, rect))


def _write_conflict_edges(
    units: Sequence[IterUnit[Any]],
    footprints: dict[int, ChunkRect],
) -> dict[int, set[int]]:
    """The adjacency of units whose writes share a write unit anywhere.

    Keyed and valued by `unit.index`. Units targeting different output
    arrays never conflict through their footprints — but a setter's
    side-channel claims (`extra_write_footprints`, if it declares any) join
    the graph too, exactly like footprints on a shared array.
    """
    adjacency: dict[int, set[int]] = {index: set() for index in footprints}
    # Grouped by *stored* array (store + path), not handle identity: two
    # `zarr.Array` objects onto the same array (e.g. from two `get_label`
    # calls) must land in one group, or their footprints would never be
    # compared. Stores are unhashable, so match against a representative.
    by_array: list[tuple[Any, list[tuple[int, ChunkRect]]]] = []
    for unit in units:
        if unit.setter is None or unit.index not in footprints:
            continue
        entry = (unit.index, footprints[unit.index])
        for representative, group in by_array:
            if _is_same_zarr_array(unit.setter.zarr_array, representative):
                group.append(entry)
                break
        else:
            by_array.append((unit.setter.zarr_array, [entry]))

    for _, group in by_array:
        _sweep_adjacency(group, adjacency)
    for group in _collect_extra_claims(units).values():
        _sweep_adjacency(group, adjacency)
    return adjacency


def plan_waves(
    units: Sequence[IterUnit[T]], *, log: bool = True
) -> list[list[IterUnit[T]]]:
    """Partition units into conflict-free waves by first-fit greedy colouring.

    Two units whose write footprints share a write unit (a chunk, or a shard
    when the output is sharded) of the same output array never share a wave:
    a wave's writes are pairwise disjoint at write granularity, so running
    one wave at a time preserves the single-writer-per-write-unit invariant
    that makes parallel writes lock-free in any topology. Read-only units and
    units with an empty write selection conflict with nothing and land in the
    first wave. Coloring runs in ascending `unit.index`, so the schedule is a
    pure function of the unit sequence. A conflict-free set — a
    `by_write_units()` tiling, say — yields a single wave.

    Flattened wave order is the canonical write order for every mapper
    (`canonical_unit_order`), so units whose writes overlap at the *pixel*
    level land in the same order serially and in parallel: the later wave
    wins, deterministically.

    Args:
        units: The units to schedule.
        log: Emit schedule-quality log messages. The parallel mappers keep
            this on; serial callers ordering their units pass `False`.
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

    if not log:
        return waves
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


def canonical_unit_order(units: Sequence[IterUnit[T]]) -> list[IterUnit[T]]:
    """Units in flattened wave order — the one write order every mapper uses.

    Wave 0 in index order, then wave 1, and so on. With no write conflicts
    this is plain index order, so for disjoint tilings it changes nothing;
    with pixel-overlapping writes it makes "who wrote last" a pure function
    of the unit sequence, identical for serial and parallel runs.
    """
    return [unit for wave in plan_waves(units, log=False) for unit in wave]


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
    That is the property `for_job` builds on — a component never spans
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

    Units run in conflict-free waves, back to back on one pool — see
    `plan_waves`. With one unit, or a resolved pool of one, this is exactly
    `BasicMapper`.

    Args:
        max_workers: `"auto"` (the default) sizes the pool for
            round-trip-bound work; an `int` pins it. `None` is accepted and
            means `"auto"`.
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
        # serialise every ROI through IPC for a result `map` discards.
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
    `None` (which `map` discards anyway); only `reduce` results are
    pickled back to the parent.

    Units run in conflict-free waves (see `plan_waves`) — which is the
    entire cross-process safety argument: no lock ngio could take would
    work across processes anyway.

    Constraints:
    - `func` (and any transforms the units carry) must be picklable — a
      module-level function, not a lambda or closure.
    - In-memory stores are refused: a `MemoryStore` pickles by value, so a
      child would write into its own copy and the parent would never see it.
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
