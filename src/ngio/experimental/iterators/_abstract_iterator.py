from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Generic, Literal, Self, TypeVar, overload

from ngio import Roi
from ngio.experimental.iterators._mappers import BasicMapper, MapperProtocol
from ngio.experimental.iterators._rois_utils import (
    by_chunks,
    by_yx,
    by_zyx,
    grid,
    rois_product,
)
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._ops_slices_utils import check_if_regions_overlap
from ngio.tables import GenericRoiTable
from ngio.utils import NgioValueError

NumpyPipeType = TypeVar("NumpyPipeType")
DaskPipeType = TypeVar("DaskPipeType")


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

    def by_chunks(self, overlap_xy: int = 0, overlap_z: int = 0) -> Self:
        """Return a new iterator that iterates over ROIs by chunks.

        Args:
            overlap_xy (int): Overlap in XY dimensions.
            overlap_z (int): Overlap in Z dimension.

        Returns:
            SegmentationIterator: A new iterator with chunked ROIs.
        """
        rois = by_chunks(
            self.rois, self.ref_image, overlap_xy=overlap_xy, overlap_z=overlap_z
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
        data_mode: Literal["numpy", "dask"] = "dask",
        iterator_mode: Literal["readwrite", "readonly"] = "readwrite",
    ) -> Generator:
        """Create an iterator over the pixels of the ROIs."""
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

    def iter_as_numpy(
        self,
    ):
        """Create an iterator over the pixels of the ROIs."""
        return self.iter(lazy=False, data_mode="numpy", iterator_mode="readwrite")

    def iter_as_dask(
        self,
    ):
        """Create an iterator over the pixels of the ROIs."""
        return self.iter(lazy=False, data_mode="dask", iterator_mode="readwrite")

    def map_as_numpy(
        self,
        func: Callable[[NumpyPipeType], NumpyPipeType],
        mapper: MapperProtocol[NumpyPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels."""
        if mapper is None:
            _mapper = BasicMapper[NumpyPipeType]()
        else:
            _mapper = mapper

        _mapper(
            func=func,
            getters=self._numpy_getters_generator(),
            setters=self._numpy_setters_generator(),
        )
        self.post_consolidate()

    def map_as_dask(
        self,
        func: Callable[[DaskPipeType], DaskPipeType],
        mapper: MapperProtocol[DaskPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels."""
        if mapper is None:
            _mapper = BasicMapper[DaskPipeType]()
        else:
            _mapper = mapper

        _mapper(
            func=func,
            getters=self._dask_getters_generator(),
            setters=self._dask_setters_generator(),
        )
        self.post_consolidate()

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
        """Check if any of the ROIs overlap in terms of chunks.

        If two ROIs cover the same chunk, they are considered to overlap in chunks.
        This does not consider pixel-level overlaps.

        Returns:
            bool: True if any ROIs overlap in chunks, False otherwise.
        """
        from ngio.io_pipes._ops_slices_utils import check_if_chunks_overlap

        if len(self.rois) < 2:
            # Less than 2 ROIs cannot overlap
            return False

        slicing_tuples = (
            g.slicing_ops.normalized_slicing_tuple
            for g in self._numpy_getters_generator()
        )
        shape = self.ref_image.shape
        chunks = self.ref_image.chunks
        return check_if_chunks_overlap(slicing_tuples, shape, chunks)

    def require_no_chunks_overlap(self) -> None:
        """Ensure that the ROIs do not overlap in terms of chunks."""
        if self.check_if_chunks_overlap():
            raise NgioValueError("Some rois overlap in chunks.")
