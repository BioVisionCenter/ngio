"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
from pydantic import BaseModel, model_validator
from zarr.core.array import CompressorLike

from ngio.common import (
    Dimensions,
    InterpolationOrder,
    Roi,
)
from ngio.common._pyramid import ChunksLike, ShardsLike
from ngio.images._abstract_image import AbstractImage, abstract_derive
from ngio.io_pipes import (
    SlicingInputType,
    TransformProtocol,
)
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
)
from ngio.utils import (
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class ChannelSelectionModel(BaseModel):
    """Model for channel selection.

    This model is used to select a channel by label, wavelength ID, or index.

    Args:
        identifier (str): Unique identifier for the channel.
            This can be a channel label, wavelength ID, or index.
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer).

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifier: str

    @model_validator(mode="after")
    def check_channel_selection(self):
        if self.mode == "index":
            try:
                int(self.identifier)
            except ValueError as e:
                raise ValueError(
                    "Identifier must be an integer when mode is 'index'"
                ) from e
        return self


ChannelSlicingInputType = (
    None
    | int
    | str
    | ChannelSelectionModel
    | Sequence[str | ChannelSelectionModel | int]
)


def _check_channel_meta(meta: NgioImageMeta, dimension: Dimensions) -> ChannelsMeta:
    """Check the channel metadata."""
    c_dim = dimension.get("c", default=1)

    if meta.channels_meta is None:
        return ChannelsMeta.default_init(labels=c_dim)

    if len(meta.channels_meta.channels) != c_dim:
        raise NgioValueError(
            "The number of channels does not match the image. "
            f"Expected {len(meta.channels_meta.channels)} channels, got {c_dim}."
        )

    return meta.channels_meta


class Image(AbstractImage):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = ImageMetaHandler(group_handler)
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    @property
    def meta_handler(self) -> ImageMetaHandler:
        """Return the metadata handler."""
        assert isinstance(self._meta_handler, ImageMetaHandler)
        return self._meta_handler

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        meta = self.meta_handler.get_meta()
        assert isinstance(meta, NgioImageMeta)
        return meta

    @property
    def channels_meta(self) -> ChannelsMeta:
        """Return the channels metadata."""
        return _check_channel_meta(self.meta, self.dimensions)

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        return self.channels_meta.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the list of wavelength of the image."""
        return self.channels_meta.channel_wavelength_ids

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return len(self.channel_labels)

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by its label or wavelength ID."""
        return self.channels_meta.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def get_as_numpy(
        self,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: slice | int | Sequence[int] | None,
    ) -> np.ndarray:
        """Get the image as a numpy array.

        Args:
            channel_selection: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_as_numpy(
            axes_order=axes_order, transforms=transforms, **_slicing_kwargs
        )

    def get_roi_as_numpy(
        self,
        roi: Roi,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Get the image as a numpy array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_selection: Select a what subset of channels to return.
                If None, all channels are returned.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_roi_as_numpy(
            roi=roi, axes_order=axes_order, transforms=transforms, **_slicing_kwargs
        )

    def get_as_dask(
        self,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get the image as a dask array.

        Args:
            channel_selection: Select a what subset of channels to return.
                If None, all channels are returned.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The dask array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_as_dask(
            axes_order=axes_order, transforms=transforms, **_slicing_kwargs
        )

    def get_roi_as_dask(
        self,
        roi: Roi,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get the image as a dask array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_selection: Select a what subset of channels to return.
                If None, all channels are returned.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The dask array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_roi_as_dask(
            roi=roi, axes_order=axes_order, transforms=transforms, **_slicing_kwargs
        )

    def get_array(
        self,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray | da.Array:
        """Get the image as a zarr array.

        Args:
            channel_selection: Select a what subset of channels to return.
                If None, all channels are returned.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return.
                Can be "dask", "numpy".
            **slicing_kwargs: The slices to get the array.

        Returns:
            The zarr array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_array(
            axes_order=axes_order, mode=mode, transforms=transforms, **_slicing_kwargs
        )

    def get_roi(
        self,
        roi: Roi,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray | da.Array:
        """Get the image as a zarr array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_selection: Select a what subset of channels to return.
                If None, all channels are returned.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return.
                Can be "dask", "numpy".
            **slicing_kwargs: The slices to get the array.

        Returns:
            The zarr array of the region of interest.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        return self._get_roi(
            roi=roi,
            axes_order=axes_order,
            mode=mode,
            transforms=transforms,
            **_slicing_kwargs,
        )

    def set_array(
        self,
        patch: np.ndarray | da.Array,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set the image array.

        Args:
            patch: The array to set.
            channel_selection: Select a what subset of channels to return.
                If None, all channels are set.
            axes_order: The order of the axes to set the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to set the array.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        self._set_array(
            patch=patch, axes_order=axes_order, transforms=transforms, **_slicing_kwargs
        )

    def set_roi(
        self,
        roi: Roi,
        patch: np.ndarray | da.Array,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set the image array for a region of interest.

        Args:
            roi: The region of interest to set the array.
            patch: The array to set.
            channel_selection: Select a what subset of channels to return.
            axes_order: The order of the axes to set the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to set the array.
        """
        _slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self, channel_selection=channel_selection, slicing_dict=slicing_kwargs
        )
        self._set_roi(
            roi=roi,
            patch=patch,
            axes_order=axes_order,
            transforms=transforms,
            **_slicing_kwargs,
        )

    def consolidate(
        self,
        order: InterpolationOrder = "linear",
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        self._consolidate(order=order, mode=mode)


class ImagesContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        self._meta_handler = ImageMetaHandler(group_handler)

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        return self._meta_handler.get_meta()

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._meta_handler.get_meta().levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._meta_handler.get_meta().paths

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        image = self.get()
        return image.num_channels

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        image = self.get()
        return image.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the wavelength of the image."""
        image = self.get()
        return image.wavelength_ids

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by label or wavelength ID.

        Args:
            channel_label (str | None): The label of the channel.
                If None a wavelength ID must be provided.
            wavelength_id (str | None): The wavelength ID of the channel.
                If None a channel label must be provided.

        Returns:
            int: The index of the channel.

        """
        image = self.get()
        return image.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def _set_channel_meta(
        self,
        channels_meta: ChannelsMeta,
    ) -> None:
        """Set the channels metadata."""
        meta = self.meta
        meta.set_channels_meta(channels_meta)
        self._meta_handler.update_meta(meta)

    def set_channel_meta(
        self,
        labels: Sequence[str | None] | int | None = None,
        wavelength_id: Sequence[str | None] | None = None,
        start: Sequence[float | None] | None = None,
        end: Sequence[float | None] | None = None,
        percentiles: tuple[float, float] | None = None,
        colors: Sequence[str | None] | None = None,
        active: Sequence[bool | None] | None = None,
        **omero_kwargs: dict,
    ) -> None:
        """Create a ChannelsMeta object with the default unit.

        Args:
            labels(Sequence[str | None] | int): The list of channels names
                in the image. If an integer is provided, the channels will
                be named "channel_i".
            wavelength_id(Sequence[str | None]): The wavelength ID of the channel.
                If None, the wavelength ID will be the same as the channel name.
            start(Sequence[float | None]): The start value for each channel.
                If None, the start value will be computed from the image.
            end(Sequence[float | None]): The end value for each channel.
                If None, the end value will be computed from the image.
            percentiles(tuple[float, float] | None): The start and end
                percentiles for each channel. If None, the percentiles will
                not be computed.
            colors(Sequence[str | None]): The list of colors for the
                channels. If None, the colors will be random.
            active (Sequence[bool | None]): Whether the channel should
                be shown by default.
            omero_kwargs(dict): Extra fields to store in the omero attributes.
        """
        low_res_dataset = self.meta.get_lowest_resolution_dataset()
        ref_image = self.get(path=low_res_dataset.path)

        if start is not None and end is None:
            raise NgioValueError("If start is provided, end must be provided as well.")
        if end is not None and start is None:
            raise NgioValueError("If end is provided, start must be provided as well.")

        if start is not None and percentiles is not None:
            raise NgioValueError(
                "If start and end are provided, percentiles must be None."
            )

        if percentiles is not None:
            start, end = compute_image_percentile(
                ref_image,
                start_percentile=percentiles[0],
                end_percentile=percentiles[1],
            )
        elif start is not None and end is not None:
            if len(start) != len(end):
                raise NgioValueError(
                    "The start and end lists must have the same length."
                )
            if len(start) != self.num_channels:
                raise NgioValueError(
                    "The start and end lists must have the same length as "
                    "the number of channels."
                )

            start = list(start)
            end = list(end)

        else:
            start, end = None, None

        if labels is None:
            labels = ref_image.num_channels

        channel_meta = ChannelsMeta.default_init(
            labels=labels,
            wavelength_id=wavelength_id,
            colors=colors,
            start=start,
            end=end,
            active=active,
            data_type=ref_image.dtype,
            **omero_kwargs,
        )
        self._set_channel_meta(channel_meta)

    def set_channel_percentiles(
        self,
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
    ) -> None:
        """Update the percentiles of the channels."""
        if self.meta._channels_meta is None:
            raise NgioValueError("The channels meta is not initialized.")

        low_res_dataset = self.meta.get_lowest_resolution_dataset()
        ref_image = self.get(path=low_res_dataset.path)
        starts, ends = compute_image_percentile(
            ref_image, start_percentile=start_percentile, end_percentile=end_percentile
        )

        channels = []
        for c, channel in enumerate(self.meta._channels_meta.channels):
            new_v = ChannelVisualisation(
                start=starts[c],
                end=ends[c],
                **channel.channel_visualisation.model_dump(exclude={"start", "end"}),
            )
            new_c = Channel(
                channel_visualisation=new_v,
                **channel.model_dump(exclude={"channel_visualisation"}),
            )
            channels.append(new_c)

        new_meta = ChannelsMeta(channels=channels)

        meta = self.meta
        meta.set_channels_meta(new_meta)
        self._meta_handler.update_meta(meta)

    def set_axes_unit(
        self,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
    ) -> None:
        """Set the axes unit of the image.

        Args:
            space_unit (SpaceUnits): The space unit of the image.
            time_unit (TimeUnits): The time unit of the image.
        """
        meta = self.meta
        meta = meta.to_units(space_unit=space_unit, time_unit=time_unit)
        self._meta_handler.update_meta(meta)

    def derive(
        self,
        store: StoreOrGroup,
        ref_path: str | None = None,
        # Metadata parameters
        shape: Sequence[int] | None = None,
        pixelsize: float | tuple[float, float] | None = None,
        z_spacing: float | None = None,
        time_spacing: float | None = None,
        name: str | None = None,
        channels_meta: Sequence[str | Channel] | None = None,
        channels_policy: Literal["same", "squeeze", "singleton"] | int = "same",
        ngff_version: NgffVersions | None = None,
        # Zarr Array parameters
        chunks: ChunksLike | None = None,
        shards: ShardsLike | None = None,
        dtype: str = "uint16",
        dimension_separator: Literal[".", "/"] = "/",
        compressors: CompressorLike = "auto",
        extra_array_kwargs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        # Deprecated arguments
        labels: Sequence[str] | None = None,
        pixel_size: PixelSize | None = None,
    ) -> "ImagesContainer":
        """Create an empty OME-Zarr image from an existing image.

        If a kwarg is not provided, the value from the reference image will be used.

        Args:
            image_container (ImagesContainer): The image container to derive the new
                image.
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_path (str | None): The path to the reference image in the image
                container.
            shape (Sequence[int] | None): The shape of the new image.
            pixelsize (float | tuple[float, float] | None): The pixel size of the new
                image.
            z_spacing (float | None): The z spacing of the new image.
            time_spacing (float | None): The time spacing of the new image.
            name (str | None): The name of the new image.
            channels_meta (Sequence[str | Channel] | None): The channels metadata
                of the new image.
            channels_policy (Literal["same", "squeeze", "singleton"] | int):
                Possible policies:
                - If "squeeze", the channels axis will be removed (no matter its size).
                - If "same", the channels axis will be kept as is (if it exists).
                - If "singleton", the channels axis will be set to size 1.
                - If an integer is provided, the channels axis will be changed to have
                    that size.
            ngff_version (NgffVersions | None): The NGFF version to use.
            chunks (ChunksLike | None): The chunk shape of the new image.
            shards (ShardsLike | None): The shard shape of the new image.
            dtype (str | None): The data type of the new image.
            dimension_separator (DIMENSION_SEPARATOR | None): The separator to use for
                dimensions.
            compressors (CompressorLike | None): The compressors to use.
            extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
                the zarr array creation.
            overwrite (bool): Whether to overwrite an existing image.
            labels (Sequence[str] | None): The labels of the new image.
                This argument is deprecated please use channels_meta instead.
            pixel_size (PixelSize | None): The pixel size of the new image.
                This argument is deprecated please use pixelsize, z_spacing,
                and time_spacing instead.

        Returns:
            ImagesContainer: The new derived image.

        """
        return derive_image_container(
            image_container=self,
            store=store,
            ref_path=ref_path,
            shape=shape,
            pixelsize=pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            name=name,
            channels_meta=channels_meta,
            channels_policy=channels_policy,
            ngff_version=ngff_version,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            dimension_separator=dimension_separator,
            compressors=compressors,
            extra_array_kwargs=extra_array_kwargs,
            overwrite=overwrite,
            labels=labels,
            pixel_size=pixel_size,
        )

    def get(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Image:
        """Get an image at a specific level.

        Args:
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size (PixelSize | None): The pixel size of the image.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.

        """
        dataset = self._meta_handler.get_meta().get_dataset(
            path=path, pixel_size=pixel_size, strict=strict
        )
        return Image(
            group_handler=self._group_handler,
            path=dataset.path,
            meta_handler=self._meta_handler,
        )


def compute_image_percentile(
    image: Image,
    start_percentile: float = 0.1,
    end_percentile: float = 99.9,
) -> tuple[list[float], list[float]]:
    """Compute the start and end percentiles for each channel of an image.

    Args:
        image: The image to compute the percentiles for.
        start_percentile: The start percentile to compute.
        end_percentile: The end percentile to compute.

    Returns:
        A tuple containing the start and end percentiles for each channel.
    """
    starts, ends = [], []
    for c in range(image.num_channels):
        if image.num_channels == 1:
            data = image.get_as_dask()
        else:
            data = image.get_as_dask(c=c)

        data = da.ravel(data)
        # remove all the zeros
        mask = data > 1e-16
        data = data[mask]
        _data = data.compute()
        if _data.size == 0:
            starts.append(0.0)
            ends.append(0.0)
            continue

        # compute the percentiles
        _s_perc, _e_perc = da.percentile(
            data, [start_percentile, end_percentile], method="nearest"
        ).compute()  # type: ignore (return type is a tuple of floats)

        starts.append(float(_s_perc))
        ends.append(float(_e_perc))
    return starts, ends


def derive_image_container(
    *,
    image_container: ImagesContainer,
    store: StoreOrGroup,
    ref_path: str | None = None,
    # Metadata parameters
    shape: Sequence[int] | None = None,
    pixelsize: float | tuple[float, float] | None = None,
    z_spacing: float | None = None,
    time_spacing: float | None = None,
    name: str | None = None,
    channels_policy: Literal["same", "squeeze", "singleton"] | int = "same",
    channels_meta: Sequence[str | Channel] | None = None,
    ngff_version: NgffVersions | None = None,
    # Zarr Array parameters
    chunks: ChunksLike | None = None,
    shards: ShardsLike | None = None,
    dtype: str | None = None,
    dimension_separator: Literal[".", "/"] | None = None,
    compressors: CompressorLike | None = None,
    extra_array_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    # Deprecated arguments
    labels: Sequence[str] | None = None,
    pixel_size: PixelSize | None = None,
) -> ImagesContainer:
    """Derive a new OME-Zarr image container from an existing image.

    If a kwarg is not provided, the value from the reference image will be used.

    Args:
        image_container (ImagesContainer): The image container to derive the new image
            from.
        store (StoreOrGroup): The Zarr store or group to create the image in.
        ref_path (str | None): The path to the reference image in the image container.
        shape (Sequence[int] | None): The shape of the new image.
        pixelsize (float | tuple[float, float] | None): The pixel size of the new image.
        z_spacing (float | None): The z spacing of the new image.
        time_spacing (float | None): The time spacing of the new image.
        name (str | None): The name of the new image.
        channels_policy (Literal["squeeze", "same", "singleton"] | int): Possible
            policies:
            - If "squeeze", the channels axis will be removed (no matter its size).
            - If "same", the channels axis will be kept as is (if it exists).
            - If "singleton", the channels axis will be set to size 1.
            - If an integer is provided, the channels axis will be changed to have
                that size.
        channels_meta (Sequence[str | Channel] | None): The channels metadata
            of the new image.
        ngff_version (NgffVersions | None): The NGFF version to use.
        chunks (ChunksLike | None): The chunk shape of the new image.
        shards (ShardsLike | None): The shard shape of the new image.
        dtype (str | None): The data type of the new image.
        dimension_separator (Literal[".", "/"] | None): The separator to use for
            dimensions.
        compressors (CompressorLike | None): The compressors to use.
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation.
        overwrite (bool): Whether to overwrite an existing image. Defaults to False.
        labels (Sequence[str] | None): Deprecated. This argument is deprecated,
            please use channels_meta instead.
        pixel_size (PixelSize | None): Deprecated. The pixel size of the new image.
            This argument is deprecated, please use pixelsize, z_spacing,
            and time_spacing instead.

    Returns:
        ImagesContainer: The new derived image container.

    """
    ref_image = image_container.get(path=ref_path)
    group_handler = abstract_derive(
        ref_image=ref_image,
        meta_type=NgioImageMeta,
        store=store,
        shape=shape,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        name=name,
        channels_meta=channels_meta,
        channels_policy=channels_policy,
        ngff_version=ngff_version,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        extra_array_kwargs=extra_array_kwargs,
        overwrite=overwrite,
        labels=labels,
        pixel_size=pixel_size,
    )
    return ImagesContainer(group_handler=group_handler)


def _parse_str_or_model(
    image: Image, channel_selection: int | str | ChannelSelectionModel
) -> int:
    """Parse a string or ChannelSelectionModel to an integer channel index."""
    if isinstance(channel_selection, int):
        if channel_selection < 0:
            raise NgioValueError("Channel index must be a non-negative integer.")
        if channel_selection >= image.num_channels:
            raise NgioValueError(
                "Channel index must be less than the number "
                f"of channels ({image.num_channels})."
            )
        return channel_selection
    elif isinstance(channel_selection, str):
        return image.get_channel_idx(channel_label=channel_selection)
    elif isinstance(channel_selection, ChannelSelectionModel):
        if channel_selection.mode == "label":
            return image.get_channel_idx(
                channel_label=str(channel_selection.identifier)
            )
        elif channel_selection.mode == "wavelength_id":
            return image.get_channel_idx(
                wavelength_id=str(channel_selection.identifier)
            )
        elif channel_selection.mode == "index":
            return int(channel_selection.identifier)
    raise NgioValueError(
        "Invalid channel selection type. "
        f"{channel_selection} is of type {type(channel_selection)} ",
        "supported types are str, ChannelSelectionModel, and int.",
    )


def _parse_channel_selection(
    image: Image, channel_selection: ChannelSlicingInputType
) -> dict[str, SlicingInputType]:
    """Parse the channel selection input into a list of channel indices."""
    if channel_selection is None:
        return {}
    if isinstance(channel_selection, int | str | ChannelSelectionModel):
        channel_index = _parse_str_or_model(image, channel_selection)
        return {"c": channel_index}
    elif isinstance(channel_selection, Sequence):
        _sequence = [_parse_str_or_model(image, cs) for cs in channel_selection]
        return {"c": _sequence}
    raise NgioValueError(
        f"Invalid channel selection type {type(channel_selection)}. "
        "Supported types are int, str, ChannelSelectionModel, and Sequence."
    )


def add_channel_selection_to_slicing_dict(
    image: Image,
    channel_selection: ChannelSlicingInputType,
    slicing_dict: dict[str, SlicingInputType],
) -> dict[str, SlicingInputType]:
    """Add channel selection information to the slicing dictionary."""
    channel_info = _parse_channel_selection(image, channel_selection)
    if "c" in slicing_dict and channel_info:
        raise NgioValueError(
            "Both channel_selection and 'c' in slicing_kwargs are provided. "
            "Which channel selection should be used is ambiguous. "
            "Please provide only one."
        )
    slicing_dict = slicing_dict | channel_info
    return slicing_dict
