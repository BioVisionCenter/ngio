"""Mappers for iterators.

Mappers execute a function over the units produced by an iterator and collect
the results. They can be passed to the `map_as_*`/`reduce_as_*` methods of
iterators to customize how the units are scheduled.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from ngio.common import Roi
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._ops_slices_utils import ChunkRect, compute_chunk_rect

T = TypeVar("T")
R = TypeVar("R")


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


@dataclass(frozen=True)
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
    - The returned list is ordered by `unit.index` (`results[i]` corresponds
      to `iterator.rois[i]`), regardless of execution order.
    - `units` is typically a generator and each unit is expensive to build
      (store metadata round-trips); generators are not thread-safe. A parallel
      mapper must materialise and consume `units` on the dispatching thread
      only, then distribute the materialised units to its workers.
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
                # reduce_as_* always builds units with setter=None.
                unit.setter(cast("T", result))
            results.append((unit.index, result))
        results.sort(key=lambda item: item[0])
        return [result for _, result in results]
