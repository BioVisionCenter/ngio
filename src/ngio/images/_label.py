"""A module for handling label images in OME-NGFF files."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
from zarr.core.array import CompressorLike

from ngio.common import Roi, compute_masking_roi
from ngio.common._label_ops import relabel_sequential
from ngio.common._pyramid import (
    ChunksLike,
    ConsolidationMode,
    ShardsLike,
)
from ngio.images._abstract_image import (
    AbstractImage,
    ConsolidationRegions,
    abstract_derive,
)
from ngio.images._image import Image
from ngio.io_pipes import MergeInput, SlicingInputType, TransformProtocol
from ngio.ome_zarr_meta import (
    LabelMetaHandler,
    LabelsGroupMetaHandler,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    PixelSize,
    update_ngio_labels_group_meta,
)
from ngio.ome_zarr_meta.ngio_specs import (
    NgffVersions,
)
from ngio.ome_zarr_meta.ngio_specs._axes import AxesSetup
from ngio.tables import MaskingRoiTable
from ngio.utils import (
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class Label(AbstractImage):
    """A single level of a label pyramid."""

    def get_as_numpy(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Get the label as a numpy array.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.
        """
        return self._get_as_numpy(
            axes_order=axes_order, transforms=transforms, **slicing_kwargs
        )

    def get_as_dask(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get the label as a dask array.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.
        """
        return self._get_as_dask(
            axes_order=axes_order, transforms=transforms, **slicing_kwargs
        )

    def get_array(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray | da.Array:
        """Get the label as a numpy or dask array, by `mode`.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return ("numpy" or "dask").
            **slicing_kwargs: The slices to get the array.
        """
        return self._get_array(
            axes_order=axes_order, transforms=transforms, mode=mode, **slicing_kwargs
        )

    def get_roi_as_numpy(
        self,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Get a region of the label as a numpy array.

        Args:
            roi: The region of interest to get.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: Per-axis selections in absolute coordinates; an
                explicit selection on an axis the `roi` already pins replaces
                the roi-derived one (and drops the pipe's `roi`).
        """
        return self._get_roi_as_numpy(
            roi, axes_order=axes_order, transforms=transforms, **slicing_kwargs
        )

    def get_roi_as_dask(
        self,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get a region of the label as a dask array.

        Args:
            roi: The region of interest to get.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: Per-axis selections in absolute coordinates; an
                explicit selection on an axis the `roi` already pins replaces
                the roi-derived one (and drops the pipe's `roi`).
        """
        return self._get_roi_as_dask(
            roi, axes_order=axes_order, transforms=transforms, **slicing_kwargs
        )

    def get_roi(
        self,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray | da.Array:
        """Get a region of the label as a numpy or dask array, by `mode`.

        Args:
            roi: The region of interest to get.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return ("numpy" or "dask").
            **slicing_kwargs: Per-axis selections in absolute coordinates; an
                explicit selection on an axis the `roi` already pins replaces
                the roi-derived one (and drops the pipe's `roi`).
        """
        return self._get_roi(
            roi,
            axes_order=axes_order,
            transforms=transforms,
            mode=mode,
            **slicing_kwargs,
        )

    def set_array(
        self,
        patch: np.ndarray | da.Array,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        merge: MergeInput | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Write a patch to the label.

        Dask patches are serial-only: concurrent dask writes from several
        threads can silently lose updates (numpy patches are unaffected).

        Args:
            patch: The patch to set.
            axes_order: The order of the axes of the patch.
            transforms: The transforms to apply to the patch.
            merge: How to combine the patch with what is already there.
                `None` overwrites. See `ngio.transforms`.
            **slicing_kwargs: The slices to set the patch.
        """
        return self._set_array(
            patch,
            axes_order=axes_order,
            transforms=transforms,
            merge=merge,
            **slicing_kwargs,
        )

    def set_roi(
        self,
        roi: Roi,
        patch: np.ndarray | da.Array,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        merge: MergeInput | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Write a patch to a region of the label.

        Dask patches are serial-only: concurrent dask writes from several
        threads can silently lose updates (numpy patches are unaffected).

        Args:
            roi: The region of interest to set.
            patch: The patch to set.
            axes_order: The order of the axes of the patch.
            transforms: The transforms to apply to the patch.
            merge: How to combine the patch with what is already there.
                `None` overwrites. See `ngio.transforms`.
            **slicing_kwargs: Per-axis selections in absolute coordinates; an
                explicit selection on an axis the `roi` already pins replaces
                the roi-derived one (and drops the pipe's `roi`).
        """
        return self._set_roi(
            roi,
            patch,
            axes_order=axes_order,
            transforms=transforms,
            merge=merge,
            **slicing_kwargs,
        )

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    def __repr__(self) -> str:
        """Return the string representation of the label."""
        return f"Label(path={self.path}, {self.dimensions})"

    @property
    def meta_handler(self) -> LabelMetaHandler:
        """Return the metadata handler."""
        assert isinstance(self._meta_handler, LabelMetaHandler)
        return self._meta_handler

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        meta = self.meta_handler.get_meta()
        assert isinstance(meta, NgioLabelMeta)
        return meta

    def build_masking_roi_table(
        self, axes_order: Sequence[str] | None = None
    ) -> MaskingRoiTable:
        """Compute the masking ROI table."""
        return build_masking_roi_table(self, axes_order=axes_order)

    def consolidate(
        self,
        mode: ConsolidationMode | None = None,
        regions: ConsolidationRegions | None = None,
    ) -> None:
        """Consolidate the label on disk.

        Args:
            mode: How to build each level, see `ConsolidationMode`.
            regions: Where this level changed — `Roi`s or on-disk index
                tuples — to rebuild only what derives from it. See
                `Image.consolidate`.
        """
        self._consolidate(
            order="nearest",
            mode=mode,
            regions=regions,
        )

    def relabel_sequential(
        self,
        consolidation_mode: ConsolidationMode | None = None,
    ) -> int:
        """Renumber the objects to a dense `1..N`, in place.

        Useful after any process that leaves gaps in the ids — a segmentation
        written region by region, a filtering step that dropped objects, or a
        stitch run with `compact=False`.

        Numbers are handed out in first-encounter order over the chunk grid
        rather than by sorting the existing ids, which is what lets this be a
        single pass over the label instead of one pass to collect and another to
        write. Which object ends up as `1` therefore follows the array, and
        depends on the chunking.

        Args:
            consolidation_mode: How to rebuild the pyramid afterwards, see
                `consolidate`. Every level derives from level 0, so they would
                otherwise disagree with the renumbered ids.

        Returns:
            How many distinct objects the label now holds.
        """
        count = len(relabel_sequential(self.zarr_array))
        self.consolidate(mode=consolidation_mode)
        return count


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        ngff_version: NgffVersions | None = None,
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        self._axes_setup = axes_setup or AxesSetup()
        # One handler per label, shared by every `Label` handed out for that
        # name — both to decode each document once, and so that
        # `OmeZarrContainer.refresh()` can reach live `Label` objects
        # through `invalidate()`.
        self._label_meta_handlers: dict[str, LabelMetaHandler] = {}
        # If the group is empty, initialize the metadata
        try:
            self._meta_handler = LabelsGroupMetaHandler(group_handler)
        except NgioValidationError:
            if ngff_version is None:
                raise NgioValueError(
                    "The /labels group is missing metadata. "
                    "Please provide the ngff_version to initialize it."
                ) from None
            meta = NgioLabelsGroupMeta(labels=[], version=ngff_version)
            update_ngio_labels_group_meta(
                group_handler=group_handler,
                ngio_meta=meta,
            )
            self._group_handler = self._group_handler.reopen_handler()
            self._meta_handler = LabelsGroupMetaHandler(group_handler)

    @property
    def meta(self) -> NgioLabelsGroupMeta:
        """Return the metadata."""
        meta = self._meta_handler.get_meta()
        return meta

    @property
    def axes_setup(self) -> AxesSetup:
        """Return the axes setup."""
        return self._axes_setup

    def list(self) -> list[str]:
        """Return the list of label names in the group."""
        return self.meta.labels

    def get(
        self,
        name: str,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Label:
        """Get a label from the group.

        Args:
            name (str): The name of the label.
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size: Select the pyramid level whose pixel size matches this one.
                A lookup key, not a value to write; to set a pixel size see
                `pixelsize` on the create/derive entry points.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.

        """
        if name not in self.list():
            raise NgioValueError(
                f"Label '{name}' not found in the Labels group. "
                f"Available labels: {self.list()}"
            )

        group_handler = self._group_handler.get_handler(name)
        label_meta_handler = self._label_meta_handlers.get(name)
        if label_meta_handler is None:
            label_meta_handler = LabelMetaHandler(
                group_handler, axes_setup=self.axes_setup
            )
            # `setdefault`: concurrent getters of one label must share the
            # winner, or a later `invalidate()` would miss the losers.
            label_meta_handler = self._label_meta_handlers.setdefault(
                name, label_meta_handler
            )
        path = (
            label_meta_handler.get_meta()
            .get_dataset(path=path, pixel_size=pixel_size, strict=strict)
            .path
        )
        return Label(
            group_handler=group_handler,
            path=path,
            meta_handler=label_meta_handler,
        )

    def invalidate(self) -> None:
        """Drop the decoded metadata of every label handed out so far.

        Moves each handler's `generation`, so live `Label` objects re-derive
        `dimensions` on their next access instead of serving a snapshot.
        """
        for handler in self._label_meta_handlers.values():
            handler.invalidate()

    def delete(self, name: str, missing_ok: bool = False) -> None:
        """Delete a label from the group.

        Args:
            name (str): The name of the label to delete.
            missing_ok (bool): If True, do not raise an error if the label does not
                exist.

        """
        existing_labels = self.list()
        if name not in existing_labels:
            if missing_ok:
                return
            raise NgioValueError(
                f"Label '{name}' not found in the Labels group. "
                f"Available labels: {existing_labels}"
            )

        self._group_handler.delete_group(name)
        self._label_meta_handlers.pop(name, None)
        existing_labels.remove(name)
        update_meta = NgioLabelsGroupMeta(
            labels=existing_labels, version=self.meta.version
        )
        self._meta_handler.update_meta(update_meta)

    def derive(
        self,
        name: str,
        ref_image: Image | Label,
        # Metadata parameters
        shape: Sequence[int] | None = None,
        pixelsize: float | tuple[float, float] | None = None,
        z_spacing: float | None = None,
        time_spacing: float | None = None,
        translation: Sequence[float] | None = None,
        channels_policy: Literal["same", "squeeze", "singleton"] | int = "squeeze",
        ngff_version: NgffVersions | None = None,
        # Zarr Array parameters
        chunks: ChunksLike | None = None,
        shards: ShardsLike | None = None,
        dtype: str | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        compressors: CompressorLike | None = None,
        extra_array_kwargs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "Label":
        """Create an empty OME-Zarr label from an existing image or label.

        If a kwarg is not provided, the value from the reference image will be used.

        Args:
            name (str): The name of the new label.
            ref_image (Image | Label): The reference image to derive the new label from.
            shape (Sequence[int] | None): The shape of the new label.
            pixelsize (float | tuple[float, float] | None): The pixel size of the new
                label.
                A value to write, not a lookup key; to select an existing
                level see `pixel_size` on the getters.
            z_spacing (float | None): The z spacing of the new label.
            time_spacing (float | None): The time spacing of the new label.
            translation (Sequence[float] | None): The translation for each axis
                at the highest resolution level. Defaults to None.
            channels_policy (Literal["squeeze", "same", "singleton"] | int):
                Possible policies:
                - If "squeeze", the channels axis will be removed (no matter its size).
                - If "same", the channels axis will be kept as is (if it exists).
                - If "singleton", the channels axis will be set to size 1.
                - If an integer is provided, the channels axis will be changed to have
                    that size.
            ngff_version (NgffVersions | None): The NGFF version to use.
            chunks (ChunksLike | None): The chunk shape of the new label.
            shards (ShardsLike | None): The shard shape of the new label.
            dtype (str | None): The data type of the new label.
            dimension_separator (Literal[".", "/"] | None): The separator to use for
                dimensions.
            compressors (CompressorLike | None): The compressors to use.
            extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
                the zarr array creation.
            overwrite (bool): Whether to overwrite an existing label.

        Returns:
            Label: The new derived label.

        """
        existing_labels = self.list()
        if name in existing_labels and not overwrite:
            raise NgioValueError(
                f"Label '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        label_group = self._group_handler.get_group(name, create_mode=True)
        derive_label(
            ref_image=ref_image,
            store=label_group,
            shape=shape,
            pixelsize=pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            name=name,
            translation=translation,
            channels_policy=channels_policy,
            ngff_version=ngff_version,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            dimension_separator=dimension_separator,
            compressors=compressors,
            extra_array_kwargs=extra_array_kwargs,
            overwrite=overwrite,
        )

        if name not in existing_labels:
            existing_labels.append(name)

        update_meta = NgioLabelsGroupMeta(
            labels=existing_labels, version=self.meta.version
        )
        self._meta_handler.update_meta(update_meta)
        return self.get(name)


def derive_label(
    *,
    store: StoreOrGroup,
    ref_image: Image | Label,
    # Metadata parameters
    shape: Sequence[int] | None = None,
    pixelsize: float | tuple[float, float] | None = None,
    z_spacing: float | None = None,
    time_spacing: float | None = None,
    name: str | None = None,
    translation: Sequence[float] | None = None,
    channels_policy: Literal["same", "squeeze", "singleton"] | int = "squeeze",
    ngff_version: NgffVersions | None = None,
    # Zarr Array parameters
    chunks: ChunksLike | None = None,
    shards: ShardsLike | None = None,
    dtype: str | None = None,
    dimension_separator: Literal[".", "/"] | None = None,
    compressors: CompressorLike | None = None,
    extra_array_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[ZarrGroupHandler, AxesSetup]:
    """Derive a new OME-Zarr label from an existing image or label.

    If a kwarg is not provided, the value from the reference image will be used.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the label in.
        ref_image (Image | Label): The reference image to derive the new label from.
        shape (Sequence[int] | None): The shape of the new label.
        pixelsize (float | tuple[float, float] | None): The pixel size of the new label.
            A value to write, not a lookup key; to select an existing
            level see `pixel_size` on the getters.
        z_spacing (float | None): The z spacing of the new label.
        time_spacing (float | None): The time spacing of the new label.
        name (str | None): The name of the new label.
        translation (Sequence[float] | None): The translation for each axis
            at the highest resolution level. Defaults to None.
        channels_policy (Literal["squeeze", "same", "singleton"] | int): Possible
            policies:
            - If "squeeze", the channels axis will be removed (no matter its size).
            - If "same", the channels axis will be kept as is (if it exists).
            - If "singleton", the channels axis will be set to size 1.
            - If an integer is provided, the channels axis will be changed to have that
                size.
        ngff_version (NgffVersions | None): The NGFF version to use.
        chunks (ChunksLike | None): The chunk shape of the new label.
        shards (ShardsLike | None): The shard shape of the new label.
        dtype (str | None): The data type of the new label.
        dimension_separator (Literal[".", "/"] | None): The separator to use for
            dimensions.
        compressors (CompressorLike | None): The compressors to use.
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation.
        overwrite (bool): Whether to overwrite an existing label. Defaults to False.

    Returns:
        tuple[ZarrGroupHandler, AxesSetup]: The group handler of the new label
            and the axes setup.

    """
    if dtype is None and isinstance(ref_image, Image):
        dtype = "uint32"
    group_handler, axes_setup = abstract_derive(
        ref_image=ref_image,
        meta_type=NgioLabelMeta,
        store=store,
        shape=shape,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        name=name,
        translation=translation,
        channels_meta=None,
        channels_policy=channels_policy,
        ngff_version=ngff_version,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        extra_array_kwargs=extra_array_kwargs,
        overwrite=overwrite,
    )
    return group_handler, axes_setup


def build_masking_roi_table(
    label: Label, axes_order: Sequence[str] | None = None
) -> MaskingRoiTable:
    """Compute the masking ROI table for a label.

    Args:
        label: The label to compute the masking ROI table for.
        axes_order: The order of axes for the computation. If None,
            uses the label's default axes order.

    Returns:
        A MaskingRoiTable containing ROIs for each label in the segmentation.
    """
    axes_order = axes_order or label.axes
    array = label.get_as_dask(axes_order=axes_order)
    rois = compute_masking_roi(array, label.pixel_size, axes_order=axes_order)
    return MaskingRoiTable(rois, reference_label=label.meta.name)
