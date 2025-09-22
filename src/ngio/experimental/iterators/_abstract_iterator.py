from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

from ngio import Roi
from ngio.experimental.iterators._rois_utils import (
    by_chunks,
    by_yx,
    by_zyx,
    grid,
    rois_product,
)
from ngio.images._abstract_image import AbstractImage
from ngio.tables import GenericRoiTable


class AbstractIteratorBuilder(ABC):
    """Base class for building iterators over ROIs."""

    _rois: list[Roi]
    _ref_image: AbstractImage

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regions={len(self._rois)})"

    @abstractmethod
    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
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
    def build_numpy_getter(self, roi: Roi):
        """Build a getter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_numpy_setter(self, roi: Roi):
        """Build a setter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_dask_getter(self, roi: Roi):
        """Build a Dask reader function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def build_dask_setter(self, roi: Roi):
        """Build a Dask setter function for the given ROI."""
        raise NotImplementedError

    @abstractmethod
    def post_consolidate(self) -> None:
        """Post-process the consolidated data."""
        raise NotImplementedError

    def iter_as_numpy(self):
        """Create an iterator over the pixels of the ROIs."""
        for roi in self.rois:
            data = self.build_numpy_getter(roi)()
            yield data, self.build_numpy_setter(roi)
        self.post_consolidate()

    def iter_as_dask(self):
        """Create an iterator over the pixels of the ROIs."""
        for roi in self.rois:
            data = self.build_dask_getter(roi)()
            yield data, self.build_dask_setter(roi)

    def map_as_numpy(self, func: Callable) -> None:
        """Apply a transformation function to the ROI pixels."""
        for roi in self.rois:
            data = self.build_numpy_getter(roi)
            data = func(data)
            self.build_numpy_setter(roi)
        self.post_consolidate()

    def map_as_dask(self, func: Callable) -> None:
        """Apply a transformation function to the ROI pixels."""
        for roi in self.rois:
            data = self.build_dask_getter(roi)
            data = func(data)
            self.build_dask_setter(roi)
        self.post_consolidate()
