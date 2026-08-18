import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Generic, Literal, Self, TypeVar, overload

from ngio.common import Roi
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._ops_slices_utils import (
    _pairs_stream,
    check_if_regions_overlap,
    chunk_rects_intersect,
)
from ngio.iterators._mappers import (
    BasicMapper,
    IterUnit,
    MapperProtocol,
    compute_write_footprint,
)
from ngio.iterators._rois_utils import (
    by_chunks,
    by_yx,
    by_zyx,
    grid,
    rois_product,
)
from ngio.tables import GenericRoiTable
from ngio.utils import (
    NgioDeprecationWarning,
    NgioFutureWarning,
    NgioValueError,
    deprecated,
)

NumpyPipeType = TypeVar("NumpyPipeType")
DaskPipeType = TypeVar("DaskPipeType")
R = TypeVar("R")


class AbstractIteratorBuilder(ABC, Generic[NumpyPipeType, DaskPipeType]):
    """Base class for building iterators over ROIs."""

    _rois: list[Roi]
    _ref_image: AbstractImage

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regions={len(self._rois)})"

    @abstractmethod
    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator.

        This is used to clone the iterator with the same parameters
        after every "product" operation.
        """
        pass

    @property
    def rois(self) -> list[Roi]:
        """Get the list of ROIs for the iterator."""
        return self._rois

    def _set_rois(self, rois: list[Roi]) -> None:
        """Set the list of ROIs for the iterator."""
        self._rois = rois

    @property
    def ref_image(self) -> AbstractImage:
        """Get the reference image for the iterator."""
        return self._ref_image

    @property
    def output_image(self) -> AbstractImage | None:
        """The image this iterator writes to, or `None` for a read-only iterator."""
        return None

    def _new_from_rois(self, rois: list[Roi]) -> Self:
        """Create a new instance of the iterator with a different set of ROIs."""
        init_kwargs = self.get_init_kwargs()
        new_instance = self.__class__(**init_kwargs)
        new_instance._set_rois(rois)
        return new_instance

    def grid(
        self,
        size_x: int | None = None,
        size_y: int | None = None,
        size_z: int | None = None,
        size_t: int | None = None,
        stride_x: int | None = None,
        stride_y: int | None = None,
        stride_z: int | None = None,
        stride_t: int | None = None,
        base_name: str = "",
    ) -> Self:
        """Create a grid of ROIs based on the input image dimensions."""
        rois = grid(
            rois=self.rois,
            ref_image=self.ref_image,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_t=size_t,
            stride_x=stride_x,
            stride_y=stride_y,
            stride_z=stride_z,
            stride_t=stride_t,
            base_name=base_name,
        )
        return self._new_from_rois(rois)

    def by_yx(self) -> Self:
        """Return a new iterator that iterates over ROIs by YX coordinates."""
        rois = by_yx(self.rois, self.ref_image)
        return self._new_from_rois(rois)

    def by_zyx(self, strict: bool = True) -> Self:
        """Return a new iterator that iterates over ROIs by ZYX coordinates.

        Args:
            strict (bool): If True, only iterate over ZYX if a Z axis
                is present and not of size 1.

        """
        rois = by_zyx(self.rois, self.ref_image, strict=strict)
        return self._new_from_rois(rois)

    def by_chunks(
        self,
        overlap_xy: int = 0,
        overlap_z: int = 0,
        grid: Literal["read", "write"] = "read",
    ) -> Self:
        """Return a new iterator that iterates over ROIs by storage tiles.

        Args:
            overlap_xy (int): Overlap in XY dimensions.
            overlap_z (int): Overlap in Z dimension.
            grid: `"read"` (the default) sizes the tiles by the input image's
                chunk grid, which is also what this method always did.
                `"write"` sizes them by the output image's write granularity —
                the shard shape when the output is sharded, the chunk shape
                otherwise — so the resulting ROIs pass
                `check_if_chunks_overlap` by construction, which is what a
                parallel `map` needs. Falls back to the input chunk grid when
                the iterator is read-only.

        Returns:
            A new iterator with tiled ROIs.
        """
        if grid == "write":
            grid_image = self.output_image
        elif grid == "read":
            grid_image = None
        else:
            raise NgioValueError(f"Invalid grid {grid!r}; expected 'write' or 'read'.")
        rois = by_chunks(
            self.rois,
            self.ref_image,
            overlap_xy=overlap_xy,
            overlap_z=overlap_z,
            grid_image=grid_image,
        )
        return self._new_from_rois(rois)

    def product(self, other: list[Roi] | GenericRoiTable) -> Self:
        """Cartesian product of the current ROIs with an arbitrary list of ROIs."""
        if isinstance(other, GenericRoiTable):
            other = other.rois()
        rois = rois_product(self.rois, other)
        return self._new_from_rois(rois)

    @abstractmethod
    def build_numpy_getter(self, roi: Roi) -> DataGetterProtocol[NumpyPipeType]:
        """Build a getter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_numpy_setter(self, roi: Roi) -> DataSetterProtocol[NumpyPipeType] | None:
        """Build a setter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_dask_getter(self, roi: Roi) -> DataGetterProtocol[DaskPipeType]:
        """Build a Dask reader function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_dask_setter(self, roi: Roi) -> DataSetterProtocol[DaskPipeType] | None:
        """Build a Dask setter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def post_consolidate(self) -> None:
        """Post-process the consolidated data."""
        raise NotImplementedError

    def _numpy_getters_generator(self) -> Generator[DataGetterProtocol[NumpyPipeType]]:
        """Return a list of numpy getter functions for all ROIs."""
        yield from (self.build_numpy_getter(roi) for roi in self.rois)

    def _dask_getters_generator(self) -> Generator[DataGetterProtocol[DaskPipeType]]:
        """Return a list of dask getter functions for all ROIs."""
        yield from (self.build_dask_getter(roi) for roi in self.rois)

    def _numpy_setters_generator(
        self,
    ) -> Generator[DataSetterProtocol[NumpyPipeType] | None]:
        """Return a list of numpy setter functions for all ROIs."""
        yield from (self.build_numpy_setter(roi) for roi in self.rois)

    def _dask_setters_generator(
        self,
    ) -> Generator[DataSetterProtocol[DaskPipeType] | None]:
        """Return a list of dask setter functions for all ROIs."""
        yield from (self.build_dask_setter(roi) for roi in self.rois)

    def _numpy_units_generator(
        self, with_setters: bool = True
    ) -> Generator[IterUnit[NumpyPipeType]]:
        """Yield one numpy `IterUnit` per ROI, in ROI order."""
        for index, roi in enumerate(self.rois):
            yield IterUnit(
                index=index,
                roi=roi,
                getter=self.build_numpy_getter(roi),
                setter=self.build_numpy_setter(roi) if with_setters else None,
            )

    def _dask_units_generator(
        self, with_setters: bool = True
    ) -> Generator[IterUnit[DaskPipeType]]:
        """Yield one dask `IterUnit` per ROI, in ROI order."""
        for index, roi in enumerate(self.rois):
            yield IterUnit(
                index=index,
                roi=roi,
                getter=self.build_dask_getter(roi),
                setter=self.build_dask_setter(roi) if with_setters else None,
            )

    def _read_and_write_generator(
        self,
        getters: Generator[
            DataGetterProtocol[NumpyPipeType] | DataGetterProtocol[DaskPipeType]
        ],
        setters: Generator[
            DataSetterProtocol[NumpyPipeType] | DataSetterProtocol[DaskPipeType] | None
        ],
    ) -> Generator[
        tuple[
            DataGetterProtocol[NumpyPipeType] | DataGetterProtocol[DaskPipeType],
            DataSetterProtocol[NumpyPipeType] | DataSetterProtocol[DaskPipeType],
        ]
    ]:
        """Create an iterator over the pixels of the ROIs."""
        for getter, setter in zip(getters, setters, strict=True):
            if setter is None:
                name = self.__class__.__name__
                raise NgioValueError(f"Iterator is read-only: {name}")
            yield getter, setter
        self.post_consolidate()

    def _iter(
        self,
        lazy: bool = False,
        data_mode: Literal["numpy", "dask"] = "dask",
        iterator_mode: Literal["readwrite", "readonly"] = "readwrite",
    ) -> Generator:
        """Create an iterator over the pixels of the ROIs (no deprecation warnings)."""
        if data_mode == "numpy":
            getters = self._numpy_getters_generator()
            setters = self._numpy_setters_generator()
        elif data_mode == "dask":
            getters = self._dask_getters_generator()
            setters = self._dask_setters_generator()
        else:
            raise NgioValueError(f"Invalid mode: {data_mode}")

        if iterator_mode == "readonly":
            if lazy:
                return getters
            else:
                return (getter() for getter in getters)
        if lazy:
            return self._read_and_write_generator(getters, setters)
        else:
            gen = self._read_and_write_generator(getters, setters)
            return ((getter(), setter) for getter, setter in gen)

    @overload
    def iter(
        self,
        lazy: Literal[True],
        data_mode: Literal["numpy"],
        iterator_mode: Literal["readwrite"],
    ) -> Generator[
        tuple[DataGetterProtocol[NumpyPipeType], DataSetterProtocol[NumpyPipeType]]
    ]: ...

    @overload
    def iter(
        self,
        lazy: Literal[True],
        data_mode: Literal["numpy"],
        iterator_mode: Literal["readonly"] = ...,
    ) -> Generator[DataGetterProtocol[NumpyPipeType]]: ...

    @overload
    def iter(
        self,
        lazy: Literal[True],
        data_mode: Literal["dask"],
        iterator_mode: Literal["readwrite"],
    ) -> Generator[
        tuple[DataGetterProtocol[DaskPipeType], DataSetterProtocol[DaskPipeType]]
    ]: ...

    @overload
    def iter(
        self,
        lazy: Literal[True],
        data_mode: Literal["dask"],
        iterator_mode: Literal["readonly"] = ...,
    ) -> Generator[DataGetterProtocol[DaskPipeType]]: ...

    @overload
    def iter(
        self,
        lazy: Literal[False],
        data_mode: Literal["numpy"],
        iterator_mode: Literal["readwrite"],
    ) -> Generator[tuple[NumpyPipeType, DataSetterProtocol[NumpyPipeType]]]: ...

    @overload
    def iter(
        self,
        lazy: Literal[False],
        data_mode: Literal["numpy"],
        iterator_mode: Literal["readonly"] = ...,
    ) -> Generator[NumpyPipeType]: ...

    @overload
    def iter(
        self,
        lazy: Literal[False],
        data_mode: Literal["dask"],
        iterator_mode: Literal["readwrite"],
    ) -> Generator[tuple[DaskPipeType, DataSetterProtocol[DaskPipeType]]]: ...

    @overload
    def iter(
        self,
        lazy: Literal[False],
        data_mode: Literal["dask"],
        iterator_mode: Literal["readonly"] = ...,
    ) -> Generator[DaskPipeType]: ...

    def iter(
        self,
        lazy: bool = False,
        data_mode: Literal["numpy", "dask"] | None = None,
        iterator_mode: Literal["readwrite", "readonly"] = "readwrite",
    ) -> Generator:
        """Create an iterator over the pixels of the ROIs.

        Note:
            The dask data mode is deprecated and will be removed in ngio=1.2;
            from then on `iter()` yields numpy arrays.
        """
        if data_mode is None:
            warnings.warn(
                "iter() currently defaults to data_mode='dask'; in ngio=1.2 it "
                "will yield numpy arrays and the dask mode will be removed. "
                "Pass data_mode='numpy' (or use iter_as_numpy()) to adopt the "
                "future behaviour now, or data_mode='dask' to keep the current "
                "behaviour and silence this warning.",
                NgioFutureWarning,
                stacklevel=2,
            )
            data_mode = "dask"
        elif data_mode == "dask":
            warnings.warn(
                "data_mode='dask' on iter() is deprecated and will be removed "
                "in ngio=1.2; the numpy path is the only iterator backend going "
                "forward. Use data_mode='numpy', or Image.get_as_dask() for "
                "lazy whole-region access.",
                NgioDeprecationWarning,
                stacklevel=2,
            )
        return self._iter(lazy=lazy, data_mode=data_mode, iterator_mode=iterator_mode)

    def iter_as_numpy(
        self,
    ):
        """Create an iterator over the pixels of the ROIs."""
        return self._iter(lazy=False, data_mode="numpy", iterator_mode="readwrite")

    @deprecated(
        replacement="iter_as_numpy() (or Image.get_as_dask() for a lazy array)",
        removed_in="1.2",
    )
    def iter_as_dask(
        self,
    ):
        """Create an iterator over the pixels of the ROIs.

        Deprecated: removed in ngio=1.2.
        """
        return self._iter(lazy=False, data_mode="dask", iterator_mode="readwrite")

    def _require_writable_units(
        self, units: list[IterUnit[NumpyPipeType]] | list[IterUnit[DaskPipeType]]
    ) -> None:
        """Raise if every unit is read-only: there is nothing for `map` to write."""
        if units and all(unit.setter is None for unit in units):
            name = self.__class__.__name__
            raise NgioValueError(
                f"{name} is read-only: map has nothing to write back. "
                "Use reduce (or iter) to compute without writing."
            )

    def map(
        self,
        func: Callable[[NumpyPipeType], NumpyPipeType],
        mapper: MapperProtocol[NumpyPipeType, NumpyPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels and write it back.

        Args:
            func: The transformation. Under a parallel mapper it runs on
                worker threads (or processes) and must be safe there.
            mapper: How the units are scheduled. `None` (the default) is
                serial — parallel writes stay explicit opt-in. Pass
                `ThreadedMapper()` or `ProcessMapper()` to fan out; each
                sizes its own pool from its `max_workers` argument, and both
                first refuse ROIs whose write footprints share a write unit.
        """
        if mapper is None:
            mapper = BasicMapper[NumpyPipeType, NumpyPipeType]()
        units = list(self._numpy_units_generator())
        self._require_writable_units(units)
        mapper(func, units)
        self.post_consolidate()

    def map_as_numpy(
        self,
        func: Callable[[NumpyPipeType], NumpyPipeType],
        mapper: MapperProtocol[NumpyPipeType, NumpyPipeType] | None = None,
    ) -> None:
        """Alias for `map()`."""
        return self.map(func, mapper=mapper)

    @deprecated(
        replacement="map() / map_as_numpy(mapper=...)",
        removed_in="1.2",
    )
    def map_as_dask(
        self,
        func: Callable[[DaskPipeType], DaskPipeType],
        mapper: MapperProtocol[DaskPipeType, DaskPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels and write it back.

        Deprecated: removed in ngio=1.2.

        A parallel `mapper` is pointless here: the dask pipes are already
        executed by dask's own scheduler, and stacking a thread pool on top
        of it oversubscribes rather than accelerates.
        """
        if mapper is None:
            _mapper = BasicMapper[DaskPipeType, DaskPipeType]()
        else:
            _mapper = mapper

        units = list(self._dask_units_generator())
        self._require_writable_units(units)
        _mapper(func, units)
        self.post_consolidate()

    def reduce(
        self,
        func: Callable[[NumpyPipeType], R],
        mapper: MapperProtocol[NumpyPipeType, R] | None = None,
    ) -> list[R]:
        """Apply a function to every ROI and collect the results without writing.

        Units are built read-only even on writable iterators: nothing is
        written and `post_consolidate` does not run.

        Args:
            func: The function to apply; under a parallel mapper it must be
                safe on worker threads (or processes).
            mapper: How the units are scheduled. `None` (the default) is
                serial; pass `ThreadedMapper()` or `ProcessMapper()` to fan
                out, sized by their own `max_workers` argument.

        Returns:
            One result per ROI, in ROI order: `results[i]` corresponds to
            `self.rois[i]`, regardless of the mapper's execution order.
        """
        if mapper is None:
            mapper = BasicMapper[NumpyPipeType, R]()
        return mapper(func, list(self._numpy_units_generator(with_setters=False)))

    def reduce_as_numpy(
        self,
        func: Callable[[NumpyPipeType], R],
        mapper: MapperProtocol[NumpyPipeType, R] | None = None,
    ) -> list[R]:
        """Alias for `reduce()`."""
        return self.reduce(func, mapper=mapper)

    @deprecated(
        replacement="reduce() / reduce_as_numpy(mapper=...)",
        removed_in="1.2",
    )
    def reduce_as_dask(
        self,
        func: Callable[[DaskPipeType], R],
        mapper: MapperProtocol[DaskPipeType, R] | None = None,
    ) -> list[R]:
        """Apply a function to every ROI and collect the results without writing.

        Deprecated: removed in ngio=1.2.

        Units are built read-only even on writable iterators: nothing is
        written and `post_consolidate` does not run. A parallel `mapper` is
        pointless here: the units hand `func` *lazy* dask arrays, so the
        per-unit work is graph construction, not IO — see `map_as_dask`.

        Returns:
            One result per ROI, in ROI order: `results[i]` corresponds to
            `self.rois[i]`, regardless of the mapper's execution order.
        """
        if mapper is None:
            _mapper = BasicMapper[DaskPipeType, R]()
        else:
            _mapper = mapper
        return _mapper(func, list(self._dask_units_generator(with_setters=False)))

    def check_if_regions_overlap(self) -> bool:
        """Check if any of the ROIs overlap logically.

        If two ROIs cover the same pixel, they are considered to overlap.
        This does not consider chunking or other storage details.

        Returns:
            bool: True if any ROIs overlap. False otherwise.
        """
        if len(self.rois) < 2:
            # Less than 2 ROIs cannot overlap
            return False

        slicing_tuples = (
            g.slicing_ops.normalized_slicing_tuple
            for g in self._numpy_getters_generator()
        )
        x = check_if_regions_overlap(slicing_tuples)
        return x

    def require_no_regions_overlap(self) -> None:
        """Ensure that the Iterator's ROIs do not overlap."""
        if self.check_if_regions_overlap():
            raise NgioValueError("Some rois overlap.")

    def check_if_chunks_overlap(self) -> bool:
        """Check if any two ROIs write into the same chunk (or shard) of the output.

        Measured on the write target: slicing tuples come from the setters and
        the grid is the output array's write granularity — the shard shape when
        the output is sharded (writes are atomic per shard object), the chunk
        shape otherwise. Two ROIs sharing a write unit make concurrent writes
        unsafe: the read-modify-write of that unit can lose data. A read-only
        iterator has no write hazard and always returns `False`.

        This is O(n^2) in the number of ROIs; avoid calling it repeatedly in a
        loop.

        Returns:
            bool: True if any ROIs overlap in chunks, False otherwise.
        """
        if len(self.rois) < 2:
            # Less than 2 ROIs cannot overlap
            return False

        footprints = (
            compute_write_footprint(setter)
            for setter in self._numpy_setters_generator()
            if setter is not None
        )
        non_empty = (footprint for footprint in footprints if footprint is not None)
        return any(chunk_rects_intersect(fi, fj) for fi, fj in _pairs_stream(non_empty))

    def require_no_chunks_overlap(self) -> None:
        """Ensure that the ROIs do not overlap in terms of chunks."""
        if self.check_if_chunks_overlap():
            raise NgioValueError("Some rois overlap in chunks.")
