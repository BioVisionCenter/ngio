"""A module for handling label images in OME-NGFF files."""

from collections.abc import Sequence
from typing import Literal

import dask.array as da
import numpy as np

from ngio.common import (
    ArrayLike,
    TransformProtocol,
    build_roi_masked_dask_getter,
    build_roi_masked_dask_setter,
    build_roi_masked_numpy_getter,
    build_roi_masked_numpy_setter,
)
from ngio.images._image import (
    ChannelSlicingInputType,
    Image,
    SlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.images._label import Label
from ngio.ome_zarr_meta import ImageMetaHandler, LabelMetaHandler
from ngio.tables import MaskingRoiTable
from ngio.utils import (
    ZarrGroupHandler,
)


class MaskedImage(Image):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | None,
        label: Label,
        masking_roi_table: MaskingRoiTable,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.
            label: The label image.
            masking_roi_table: The masking ROI table.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._label = label
        self._masking_roi_table = masking_roi_table

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        label_name = self._label.meta.name
        if label_name is None:
            label_name = self._masking_roi_table.reference_label
        return f"MaskedImage(path={self.path}, {self.dimensions}, {label_name})"

    def get_roi_as_numpy(  # type: ignore (this ignore the method override issue)
        self,
        label: int,
        channel_selection: ChannelSlicingInputType | None = None,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_numpy(
            roi=roi,
            channel_selection=channel_selection,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi_as_dask(  # type: ignore (this ignore the method override issue)
        self,
        label: int,
        channel_selection: ChannelSlicingInputType | None = None,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> da.Array:
        """Return the array for a given ROI as a Dask array."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_dask(
            roi=roi,
            channel_selection=channel_selection,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi(  # type: ignore (this ignore the method override issue)
        self,
        label: int,
        zoom_factor: float = 1.0,
        channel_selection: ChannelSlicingInputType | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi,
            channel_selection=channel_selection,
            axes_order=axes_order,
            transforms=transforms,
            mode=mode,
            **slicing_kwargs,
        )

    def set_roi(  # type: ignore (this ignore the method override issue)
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        channel_selection: ChannelSlicingInputType | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi,
            patch=patch,
            channel_selection=channel_selection,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi_masked_as_numpy(
        self,
        label: int,
        channel_selection: ChannelSlicingInputType | None = None,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Return the masked array for a given label as a NumPy array."""
        slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )

        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        masked_getter = build_roi_masked_numpy_getter(
            roi=roi,
            zarr_array=self.zarr_array,
            label_zarr_array=self._label.zarr_array,
            dimensions=self.dimensions,
            pixel_size=self.pixel_size,
            label_dimensions=self._label.dimensions,
            label_pixel_size=self._label.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
            allow_scaling=allow_scaling,
        )
        return masked_getter()

    def get_roi_masked_as_dask(
        self,
        label: int,
        channel_selection: ChannelSlicingInputType | None = None,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Return the masked array for a given label as a Dask array."""
        slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )

        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        masked_getter = build_roi_masked_dask_getter(
            roi=roi,
            zarr_array=self.zarr_array,
            label_zarr_array=self._label.zarr_array,
            dimensions=self.dimensions,
            pixel_size=self.pixel_size,
            label_dimensions=self._label.dimensions,
            label_pixel_size=self._label.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
            allow_scaling=allow_scaling,
        )
        return masked_getter()

    def get_roi_masked(
        self,
        label: int,
        channel_selection: ChannelSlicingInputType | None = None,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        if mode == "numpy":
            return self.get_roi_masked_as_numpy(
                label=label,
                channel_selection=channel_selection,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                allow_scaling=allow_scaling,
                **slicing_kwargs,
            )
        elif mode == "dask":
            return self.get_roi_masked_as_dask(
                label=label,
                channel_selection=channel_selection,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                allow_scaling=allow_scaling,
                **slicing_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        channel_selection: ChannelSlicingInputType | None = None,
        axes_order: Sequence[str] | None = None,
        zoom_factor: float = 1.0,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set the masked array for a given label."""
        slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )

        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        if isinstance(patch, da.Array):
            path_setter = build_roi_masked_dask_setter(
                roi=roi,
                zarr_array=self.zarr_array,
                label_zarr_array=self._label.zarr_array,
                dimensions=self.dimensions,
                pixel_size=self.pixel_size,
                label_dimensions=self._label.dimensions,
                label_pixel_size=self._label.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
                allow_scaling=allow_scaling,
            )
            path_setter(patch)
        elif isinstance(patch, np.ndarray):
            path_setter = build_roi_masked_numpy_setter(
                roi=roi,
                zarr_array=self.zarr_array,
                label_zarr_array=self._label.zarr_array,
                dimensions=self.dimensions,
                pixel_size=self.pixel_size,
                label_dimensions=self._label.dimensions,
                label_pixel_size=self._label.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
                allow_scaling=allow_scaling,
            )
            path_setter(patch)
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Expected numpy.ndarray or dask.array.Array."
            )


class MaskedLabel(Label):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler | None,
        label: Label,
        masking_roi_table: MaskingRoiTable,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.
            label: The label image.
            masking_roi_table: The masking ROI table.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._label = label
        self._masking_roi_table = masking_roi_table

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        label_name = self._label.meta.name
        if label_name is None:
            label_name = self._masking_roi_table.reference_label
        return f"MaskedLabel(path={self.path}, {self.dimensions}, {label_name})"

    def get_roi_as_numpy(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> np.ndarray:
        """Return the ROI as a NumPy array."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_numpy(
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi_as_dask(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> da.Array:
        """Return the ROI as a Dask array."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_dask(
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi,
            axes_order=axes_order,
            mode=mode,
            transforms=transforms,
            **slicing_kwargs,
        )

    def set_roi(
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi,
            patch=patch,
            axes_order=axes_order,
            transforms=transforms,
            **slicing_kwargs,
        )

    def get_roi_masked_as_numpy(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Return the masked array for a given label as a NumPy array."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        masked_getter = build_roi_masked_numpy_getter(
            roi=roi,
            zarr_array=self.zarr_array,
            label_zarr_array=self._label.zarr_array,
            dimensions=self.dimensions,
            pixel_size=self.pixel_size,
            label_dimensions=self._label.dimensions,
            label_pixel_size=self._label.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
            allow_scaling=allow_scaling,
        )
        return masked_getter()

    def get_roi_masked_as_dask(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Return the masked array for a given label as a Dask array."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        masked_getter = build_roi_masked_dask_getter(
            roi=roi,
            zarr_array=self.zarr_array,
            label_zarr_array=self._label.zarr_array,
            dimensions=self.dimensions,
            pixel_size=self.pixel_size,
            label_dimensions=self._label.dimensions,
            label_pixel_size=self._label.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
            allow_scaling=allow_scaling,
        )
        return masked_getter()

    def get_roi_masked(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Sequence[str] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        if mode == "numpy":
            return self.get_roi_masked_as_numpy(
                label=label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                allow_scaling=allow_scaling,
                **slicing_kwargs,
            )

        elif mode == "dask":
            return self.get_roi_masked_as_dask(
                label=label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                allow_scaling=allow_scaling,
                **slicing_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        axes_order: Sequence[str] | None = None,
        zoom_factor: float = 1.0,
        transforms: Sequence[TransformProtocol] | None = None,
        allow_scaling: bool = True,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set the masked array for a given label."""
        roi = self._masking_roi_table.get_label(label)
        roi = roi.zoom(zoom_factor)
        if isinstance(patch, da.Array):
            path_setter = build_roi_masked_dask_setter(
                roi=roi,
                zarr_array=self.zarr_array,
                label_zarr_array=self._label.zarr_array,
                dimensions=self.dimensions,
                pixel_size=self.pixel_size,
                label_dimensions=self._label.dimensions,
                label_pixel_size=self._label.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
                allow_scaling=allow_scaling,
            )
            path_setter(patch)
        elif isinstance(patch, np.ndarray):
            path_setter = build_roi_masked_numpy_setter(
                roi=roi,
                zarr_array=self.zarr_array,
                label_zarr_array=self._label.zarr_array,
                dimensions=self.dimensions,
                pixel_size=self.pixel_size,
                label_dimensions=self._label.dimensions,
                label_pixel_size=self._label.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
                allow_scaling=allow_scaling,
            )
            path_setter(patch)
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Expected numpy.ndarray or dask.array.Array."
            )
