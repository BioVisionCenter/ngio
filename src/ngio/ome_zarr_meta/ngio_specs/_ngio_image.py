"""Image metadata models.

This module contains the models for the image metadata.
These metadata models are not adhering to the OME standard.
But they can be built from the OME standard metadata, and the
can be converted to the OME standard.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np
from pydantic import BaseModel

from ngio.ome_zarr_meta.ngio_specs._axes import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
    canonical_axes,
)
from ngio.ome_zarr_meta.ngio_specs._channels import ChannelsMeta
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValidationError, NgioValueError

T = TypeVar("T")
NgffVersions = Literal["0.4"]
DefaultNgffVersion: Literal["0.4"] = "0.4"


class ImageLabelSource(BaseModel):
    """Image label source model."""

    version: NgffVersions
    source: dict[str, str | None]

    @classmethod
    def default_init(cls, version: NgffVersions) -> "ImageLabelSource":
        """Initialize the ImageLabelSource object."""
        return cls(version=version, source={"image": "../../"})


class AbstractNgioImageMeta:
    """Base class for ImageMeta and LabelMeta."""

    def __init__(
        self, version: NgffVersions, name: str | None, datasets: list[Dataset]
    ) -> None:
        """Initialize the ImageMeta object."""
        self._version = version
        self._name = name

        if len(datasets) == 0:
            raise NgioValidationError("At least one dataset must be provided.")

        self._datasets = datasets
        self._axes_mapper = datasets[0].axes_mapper

    def __repr__(self):
        class_name = type(self).__name__
        paths = [dataset.path for dataset in self.datasets]
        on_disk_axes = self.axes_mapper.axes_names
        return f"{class_name}(name={self.name}, datasets={paths}, axes={on_disk_axes})"

    @classmethod
    def default_init(
        cls,
        levels: int | Sequence[str],
        axes_names: Sequence[str],
        pixel_size: PixelSize,
        scaling_factors: Sequence[float] | None = None,
        name: str | None = None,
        version: NgffVersions = DefaultNgffVersion,
    ):
        """Initialize the ImageMeta object."""
        axes = canonical_axes(
            axes_names,
            space_units=pixel_size.space_unit,
            time_units=pixel_size.time_unit,
        )

        px_size_dict = pixel_size.as_dict()
        scale = [px_size_dict.get(ax.on_disk_name, 1.0) for ax in axes]
        translation = [0.0] * len(scale)

        if scaling_factors is None:
            _default = {"x": 2.0, "y": 2.0}
            scaling_factors = [_default.get(ax.on_disk_name, 1.0) for ax in axes]

        if isinstance(levels, int):
            levels = [str(i) for i in range(levels)]

        datasets = []
        for level in levels:
            dataset = Dataset(
                path=level,
                on_disk_axes=axes,
                on_disk_scale=scale,
                on_disk_translation=translation,
                allow_non_canonical_axes=False,
                strict_canonical_order=True,
            )
            datasets.append(dataset)
            scale = [s * f for s, f in zip(scale, scaling_factors, strict=True)]

        return cls(
            version=version,
            name=name,
            datasets=datasets,
        )

    def to_units(
        self,
        *,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
    ):
        """Convert the pixel size to the given units.

        Args:
            space_unit(str): The space unit to convert to.
            time_unit(str): The time unit to convert to.
        """
        new_datasets = []
        for dataset in self.datasets:
            new_dataset = dataset.to_units(space_unit=space_unit, time_unit=time_unit)
            new_datasets.append(new_dataset)

        return type(self)(
            version=self.version,
            name=self.name,
            datasets=new_datasets,
        )

    @property
    def version(self) -> NgffVersions:
        """Version of the OME-NFF metadata used to build the object."""
        return self._version  # type: ignore (version is a Literal type)

    @property
    def name(self) -> str | None:
        """Name of the image."""
        return self._name

    @property
    def datasets(self) -> list[Dataset]:
        """List of datasets in the multiscale."""
        return self._datasets

    @property
    def axes_mapper(self):
        """Return the axes mapper."""
        return self._axes_mapper

    @property
    def levels(self) -> int:
        """Number of levels in the multiscale."""
        return len(self.datasets)

    @property
    def paths(self) -> list[str]:
        """List of paths of the datasets."""
        return [dataset.path for dataset in self.datasets]

    @property
    def space_unit(self) -> str | None:
        """Get the space unit of the pixel size."""
        return self.datasets[0].pixel_size.space_unit

    @property
    def time_unit(self) -> str | None:
        """Get the time unit of the pixel size."""
        return self.datasets[0].pixel_size.time_unit

    def _get_dataset_by_path(self, path: str) -> Dataset:
        """Get a dataset by its path."""
        for dataset in self.datasets:
            if dataset.path == path:
                return dataset
        raise NgioValueError(f"Dataset with path {path} not found.")

    def _get_dataset_by_index(self, idx: int) -> Dataset:
        """Get a dataset by its index."""
        if idx < 0 or idx >= len(self.datasets):
            raise NgioValueError(f"Index {idx} out of range.")
        return self.datasets[idx]

    def _find_closest_dataset(
        self, pixel_size: PixelSize, mode: str = "any"
    ) -> Dataset | None:
        """Find the closest dataset to the given pixel size.

        Args:
            pixel_size(PixelSize): The pixel size to search for.
            mode(str): The mode to find the closest dataset.
                "any": Will find the closest dataset.
                "lr": Will find closest "lower" resolution dataset.
                "hr": Will find closest "higher" resolution
        """
        min_dist = np.inf
        closest_dataset = None

        if mode == "any":
            datasets = self.datasets
        elif mode == "lr":
            # Lower resolution means that the pixel size is larger.
            datasets = [d for d in self.datasets if d.pixel_size > pixel_size]
        elif mode == "hr":
            # Higher resolution means that the pixel size is smaller.
            datasets = [d for d in self.datasets if d.pixel_size < pixel_size]
        else:
            raise NgioValueError(f"Mode {mode} not recognized.")

        for d in datasets:
            dist = d.pixel_size.distance(pixel_size)
            if dist < min_dist:
                min_dist = dist
                closest_dataset = d

        return closest_dataset

    def _get_closest_dataset(
        self, pixel_size: PixelSize, strict: bool = False
    ) -> Dataset:
        """Get a dataset with the closest pixel size.

        Args:
            pixel_size(PixelSize): The pixel size to search for.
            strict(bool): If True, the pixel size must be exactly the same.
        """
        closest_dataset = self._find_closest_dataset(pixel_size, mode="any")

        if closest_dataset is None:
            raise NgioValueError("No dataset found.")

        if strict and closest_dataset.pixel_size != pixel_size:
            raise NgioValueError(
                "No dataset with a pixel size close enough. "
                "Best match is "
                f"{closest_dataset.path}:{closest_dataset.pixel_size}"
            )
        return closest_dataset

    def get_lowest_resolution_dataset(self) -> Dataset:
        """Get the dataset with the lowest resolution."""
        dataset = self.datasets[-1]
        while True:
            lower_res_dataset = self._find_closest_dataset(
                dataset.pixel_size, mode="lr"
            )
            if lower_res_dataset is None:
                break
            dataset = lower_res_dataset
        return dataset

    def get_highest_resolution_dataset(self) -> Dataset:
        """Get the dataset with the highest resolution."""
        dataset = self.datasets[0]
        while True:
            higher_res_dataset = self._find_closest_dataset(
                dataset.pixel_size, mode="hr"
            )
            if higher_res_dataset is None:
                break
            dataset = higher_res_dataset
        return dataset

    def get_dataset(
        self,
        *,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Dataset:
        """Get a dataset by its path, index or pixel size.

        If all arguments are None, the dataset with the highest resolution is returned.

        Args:
            path(str): The path of the dataset.
            idx(int): The index of the dataset.
            pixel_size(PixelSize): The pixel size to search for.
            strict(bool): If True, the pixel size must be exactly the same.
                If pixel_size is None, strict is ignored.
        """
        # Only one of the arguments must be provided
        if (
            sum(
                [
                    path is not None,
                    idx is not None,
                    pixel_size is not None,
                ]
            )
            > 1
        ):
            raise NgioValueError("get_dataset must receive only one argument or None.")

        if path is not None:
            return self._get_dataset_by_path(path)
        elif idx is not None:
            return self._get_dataset_by_index(idx)
        elif pixel_size is not None:
            return self._get_closest_dataset(pixel_size, strict=strict)
        else:
            return self.get_highest_resolution_dataset()

    def _get_closest_datasets(self, path: str | None = None) -> tuple[Dataset, Dataset]:
        """Get the closest datasets to a dataset."""
        dataset = self.get_dataset(path=path)
        lr_dataset = self._find_closest_dataset(dataset.pixel_size, mode="lr")
        if lr_dataset is None:
            raise NgioValueError(
                "No lower resolution dataset found. "
                "This is the lowest resolution dataset."
            )
        return dataset, lr_dataset

    def scaling_factor(self, path: str | None = None) -> list[float]:
        """Get the scaling factors from a dataset to its lower resolution."""
        if self.levels == 1:
            return [1.0] * len(self.axes_mapper.axes_names)
        dataset, lr_dataset = self._get_closest_datasets(path=path)

        scaling_factors = []
        for ax_name in self.axes_mapper.axes_names:
            s_d = dataset.get_scale(ax_name)
            s_lr_d = lr_dataset.get_scale(ax_name)
            scaling_factors.append(s_lr_d / s_d)
        return scaling_factors

    def yx_scaling(self, path: str | None = None) -> tuple[float, float]:
        """Get the scaling factor from a dataset to its lower resolution."""
        if self.levels == 1:
            return 1.0, 1.0
        dataset, lr_dataset = self._get_closest_datasets(path=path)

        if lr_dataset is None:
            raise NgioValueError(
                "No lower resolution dataset found. "
                "This is the lowest resolution dataset."
            )

        s_d = dataset.get_scale("y")
        s_lr_d = lr_dataset.get_scale("y")
        scale_y = s_lr_d / s_d

        s_d = dataset.get_scale("x")
        s_lr_d = lr_dataset.get_scale("x")
        scale_x = s_lr_d / s_d

        return scale_y, scale_x

    def z_scaling(self, path: str | None = None) -> float:
        """Get the scaling factor from a dataset to its lower resolution."""
        if self.levels == 1:
            return 1.0
        dataset, lr_dataset = self._get_closest_datasets(path=path)

        s_d = dataset.get_scale("z")
        s_lr_d = lr_dataset.get_scale("z")
        return s_lr_d / s_d


class NgioLabelMeta(AbstractNgioImageMeta):
    """Label metadata model."""

    def __init__(
        self,
        version: NgffVersions,
        name: str | None,
        datasets: list[Dataset],
        image_label: ImageLabelSource | None = None,
    ) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version, name, datasets)
        image_label = (
            ImageLabelSource.default_init(self.version)
            if image_label is None
            else image_label
        )
        assert image_label is not None
        if image_label.version != version:
            raise NgioValidationError(
                "Label image version must match the metadata version."
            )
        self._image_label = image_label

    @property
    def source_image(self) -> str | None:
        source = self._image_label.source
        if "image" not in source:
            return None

        image_path = source["image"]
        return image_path

    @property
    def image_label(self) -> ImageLabelSource:
        """Get the image label metadata."""
        return self._image_label


class NgioImageMeta(AbstractNgioImageMeta):
    """Image metadata model."""

    def __init__(
        self,
        version: NgffVersions,
        name: str | None,
        datasets: list[Dataset],
        channels: ChannelsMeta | None = None,
    ) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version=version, name=name, datasets=datasets)
        self._channels_meta = channels

    @property
    def channels_meta(self) -> ChannelsMeta | None:
        """Get the channels_meta metadata."""
        return self._channels_meta

    def set_channels_meta(self, channels_meta: ChannelsMeta) -> None:
        """Set channels_meta metadata."""
        self._channels_meta = channels_meta

    def init_channels(
        self,
        labels: list[str] | int,
        wavelength_ids: list[str] | None = None,
        colors: list[str] | None = None,
        active: list[bool] | None = None,
        start: list[int | float] | None = None,
        end: list[int | float] | None = None,
        data_type: Any = np.uint16,
    ) -> None:
        """Set the channels_meta metadata for the image.

        Args:
            labels (list[str]|int): The labels of the channels.
            wavelength_ids (list[str], optional): The wavelengths of the channels.
            colors (list[str], optional): The colors of the channels.
            adjust_window (bool, optional): Whether to adjust the window.
            start_percentile (int, optional): The start percentile.
            end_percentile (int, optional): The end percentile.
            active (list[bool], optional): Whether the channel is active.
            start (list[int | float], optional): The start value of the channel.
            end (list[int | float], optional): The end value of the channel.
            end (int): The end value of the channel.
            data_type (Any): The data type of the channel.
        """
        channels_meta = ChannelsMeta.default_init(
            labels=labels,
            wavelength_id=wavelength_ids,
            colors=colors,
            active=active,
            start=start,
            end=end,
            data_type=data_type,
        )
        self.set_channels_meta(channels_meta=channels_meta)


NgioImageLabelMeta = NgioImageMeta | NgioLabelMeta
