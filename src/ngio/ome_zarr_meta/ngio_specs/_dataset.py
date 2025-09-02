"""Fractal internal module for dataset metadata handling."""

from collections.abc import Sequence

from ngio.ome_zarr_meta.ngio_specs._axes import (
    AxesMapper,
    AxesSetup,
    Axis,
    AxisType,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
)
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValidationError


class Dataset:
    """Model for a dataset in the multiscale.

    To initialize the Dataset object, the path, the axes, scale, and translation list
    can be provided with on_disk order.
    """

    def __init__(
        self,
        *,
        # args coming from ngff specs
        path: str,
        on_disk_axes: Sequence[Axis],
        on_disk_scale: Sequence[float],
        on_disk_translation: Sequence[float] | None = None,
        # user defined args
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = False,
    ):
        """Initialize the Dataset object.

        Args:
            path (str): The path of the dataset.
            on_disk_axes (list[Axis]): The list of axes in the multiscale.
            on_disk_scale (list[float]): The list of scale transformation.
                The scale transformation must have the same length as the axes.
            on_disk_translation (list[float] | None): The list of translation.
            axes_setup (AxesSetup): The axes setup object
            allow_non_canonical_axes (bool): Allow non-canonical axes.
            strict_canonical_order (bool): Strict canonical order.
        """
        self._path = path
        self._axes_mapper = AxesMapper(
            on_disk_axes=on_disk_axes,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

        if len(on_disk_scale) != len(on_disk_axes):
            raise NgioValidationError(
                "The length of the scale transformation must be the same as the axes."
            )
        self._on_disk_scale = list(on_disk_scale)

        on_disk_translation = on_disk_translation or [0.0] * len(on_disk_axes)
        if len(on_disk_translation) != len(on_disk_axes):
            raise NgioValidationError(
                "The length of the translation must be the same as the axes."
            )
        self._on_disk_translation = list(on_disk_translation)

    def get_scale(self, axis_name: str) -> float:
        """Return the scale for a given axis."""
        idx = self._axes_mapper.get_index(axis_name)
        if idx is None:
            return 1.0
        return self._on_disk_scale[idx]

    def get_translation(self, axis_name: str) -> float:
        """Return the translation for a given axis."""
        idx = self._axes_mapper.get_index(axis_name)
        if idx is None:
            return 0.0
        return self._on_disk_translation[idx]

    @property
    def path(self) -> str:
        """Return the path of the dataset."""
        return self._path

    @property
    def space_unit(self) -> str | None:
        """Return the space unit for a given axis."""
        x_axis = self._axes_mapper.get_axis("x")
        y_axis = self._axes_mapper.get_axis("y")

        if x_axis is None or y_axis is None:
            raise NgioValidationError(
                "The dataset must have x and y axes to determine the space unit."
            )

        if x_axis.unit == y_axis.unit:
            return x_axis.unit
        else:
            raise NgioValidationError(
                "Inconsistent space units. "
                f"x={x_axis.unit} and y={y_axis.unit} should have the same unit."
            )

    @property
    def time_unit(self) -> str | None:
        """Return the time unit for a given axis."""
        t_axis = self._axes_mapper.get_axis("t")
        if t_axis is None:
            return None
        return t_axis.unit

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size for the dataset."""
        return PixelSize(
            x=self.get_scale("x"),
            y=self.get_scale("y"),
            z=self.get_scale("z"),
            t=self.get_scale("t"),
            space_unit=self.space_unit,
            time_unit=self.time_unit,
        )

    @property
    def axes_mapper(self) -> AxesMapper:
        """Return the axes mapper object."""
        return self._axes_mapper

    def to_units(
        self,
        *,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
    ) -> "Dataset":
        """Convert the pixel size to the given units.

        Args:
            space_unit(str): The space unit to convert to.
            time_unit(str): The time unit to convert to.
        """
        new_axes = []
        for ax in self.axes_mapper.axes:
            if ax.axis_type == AxisType.space:
                new_ax = Axis(
                    on_disk_name=ax.on_disk_name,
                    axis_type=ax.axis_type,
                    unit=space_unit,
                )
                new_axes.append(new_ax)
            elif ax.axis_type == AxisType.time:
                new_ax = Axis(
                    on_disk_name=ax.on_disk_name, axis_type=ax.axis_type, unit=time_unit
                )
                new_axes.append(new_ax)
            else:
                new_axes.append(ax)

        return Dataset(
            path=self.path,
            on_disk_axes=new_axes,
            on_disk_scale=self._on_disk_scale,
            on_disk_translation=self._on_disk_translation,
            axes_setup=self.axes_mapper.axes_setup,
            allow_non_canonical_axes=self.axes_mapper.allow_non_canonical_axes,
            strict_canonical_order=self.axes_mapper.strict_canonical_order,
        )
