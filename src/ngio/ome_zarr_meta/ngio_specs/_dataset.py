"""Fractal internal module for dataset metadata handling."""

from collections.abc import Sequence

from ngio.ome_zarr_meta.ngio_specs._axes import (
    AxesHandler,
)
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValidationError


class Dataset:
    """Model for a dataset in the multiscale."""

    def __init__(
        self,
        *,
        # args coming from ngff specs
        path: str,
        axes_handler: AxesHandler,
        scale: Sequence[float],
        translation: Sequence[float] | None = None,
    ):
        """Initialize the Dataset object.

        Args:
            path (str): The path of the dataset.
            axes_handler (AxesHandler): The axes handler object.
            scale (list[float]): The list of scale transformation.
                The scale transformation must have the same length as the axes.
            translation (list[float] | None): The list of translation.
                The translation must have the same length as the axes.
        """
        self._path = path
        self._axes_handler = axes_handler

        if len(scale) != len(axes_handler.axes):
            raise NgioValidationError(
                "The length of the scale transformation must be the same as the axes."
            )
        self._scale = list(scale)

        translation = translation or [0.0] * len(axes_handler.axes)
        if len(translation) != len(axes_handler.axes):
            raise NgioValidationError(
                "The length of the translation must be the same as the axes."
            )
        self._translation = list(translation)

    @property
    def path(self) -> str:
        """Return the path of the dataset."""
        return self._path

    @property
    def axes_handler(self) -> AxesHandler:
        """Return the axes handler object."""
        return self._axes_handler

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size for the dataset."""
        scale = self._scale
        pix_size_dict = {}
        # Mandatory axes: x, y
        for ax in ["x", "y"]:
            index = self.axes_handler.get_index(ax)
            assert index is not None
            pix_size_dict[ax] = scale[index]

        for ax in ["z", "t"]:
            index = self.axes_handler.get_index(ax)
            pix_size_dict[ax] = scale[index] if index is not None else 1.0

        return PixelSize(
            **pix_size_dict,
            space_unit=self.axes_handler.space_unit,
            time_unit=self.axes_handler.time_unit,
        )

    @property
    def scale(self) -> tuple[float, ...]:
        """Return the scale transformation as a tuple."""
        return tuple(self._scale)

    @property
    def translation(self) -> tuple[float, ...]:
        """Return the translation as a tuple."""
        return tuple(self._translation)
