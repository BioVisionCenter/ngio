from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, cast

import dask.array as da
import numpy as np
import pandas as pd

from ngio.common import Roi
from ngio.images import Image, Label
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.io_pipes import (
    DaskGetter,
    DataGetter,
    NumpyGetter,
    TransformProtocol,
)
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.iterators._mappers import MapperProtocol
from ngio.iterators._partials import (
    INDEX_COLUMN,
    delete_partials_root,
    merge_partial_frames,
    prepare_partials,
    validate_partials_context,
    write_partial,
)
from ngio.tables import FeatureTable, Table
from ngio.utils import NgioValueError, deprecated

NumpyPipeType: TypeAlias = tuple[np.ndarray, np.ndarray, Roi]
DaskPipeType: TypeAlias = tuple[da.Array, da.Array, Roi]

T = TypeVar("T", np.ndarray, da.Array)

#: What a per-ROI feature function may return: a DataFrame, or the cheaper
#: dict-of-columns (`{"label": [...], "area": [...]}`) that concatenates
#: without building one frame per ROI.
FeatureFuncResult: TypeAlias = pd.DataFrame | dict[str, list]


class FeatureGetter(DataGetter[tuple[T, T, Roi]], Generic[T]):
    """Pairs an image getter with a label getter over the same ROI."""

    def __init__(
        self,
        image_getter: DataGetter[T],
        label_getter: DataGetter[T],
    ) -> None:
        self._image_getter = image_getter
        self._label_getter = label_getter
        # Read at most once per getter: `get()` and the two properties used to
        # each re-run the underlying read, so the documented lazy pattern
        # (`.image` then `.label`) paid double IO.
        self._image_data: T | None = None
        self._label_data: T | None = None
        super().__init__(
            zarr_array=self._image_getter.zarr_array,
            slicing_ops=self._image_getter.slicing_ops,
            axes_ops=self._image_getter.axes_ops,
            transforms=self._image_getter.transforms,
            roi=self._image_getter.roi,
        )

    def get(self) -> tuple[T, T, Roi]:
        return self.image, self.label, self.roi

    @property
    def image(self) -> T:
        if self._image_data is None:
            self._image_data = self._image_getter()
        return self._image_data

    @property
    def label(self) -> T:
        if self._label_data is None:
            self._label_data = self._label_getter()
        return self._label_data


NumpyFeatureGetter = FeatureGetter
DaskFeatureGetter = FeatureGetter


class _UnpackedFeatureFunc:
    """Adapt `func(image, label, roi)` to the unit payload tuple.

    A class rather than a closure so it pickles by reference and the wrapped
    `func` can cross a `ProcessMapper` boundary.
    """

    def __init__(
        self, func: Callable[[np.ndarray, np.ndarray, Roi], FeatureFuncResult]
    ) -> None:
        self._func = func

    def __call__(self, data: NumpyPipeType) -> FeatureFuncResult:
        image, label, roi = data
        return self._func(image, label, roi)


def _as_feature_frame(result: FeatureFuncResult) -> pd.DataFrame:
    """One ROI's result as a DataFrame, validating the contract."""
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        return pd.DataFrame(result)
    raise NgioValueError(
        f"A feature function must return a DataFrame or a dict of "
        f"columns, got {type(result).__name__}. Pass a custom "
        "`coalesce` to handle other shapes."
    )


def _default_feature_coalesce(
    results: list[FeatureFuncResult], reference_label: str | None
) -> FeatureTable:
    """Join per-ROI results into one `FeatureTable` indexed by `label`.

    Dicts of columns are concatenated as cheaply as they were produced;
    DataFrames are concatenated as they are. Either way the rows must carry
    the object id in a `label` column (or already sit on a `label` index).
    """
    if not results:
        raise NgioValueError(
            "No per-ROI results to build a table from: the iterator has no "
            "ROIs. Check the ROI table or the tiling that produced it."
        )
    frames = [_as_feature_frame(result) for result in results]
    # ROIs with no objects contribute empty frames; concatenating those would
    # silently upcast the surviving columns (an empty `label` is float64, and
    # a float index fails the table's integer-index validation).
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        raise NgioValueError(
            "Every ROI returned zero rows: there is nothing to build a table "
            "from. Check that the label actually contains objects inside the "
            "iterated ROIs."
        )
    joined = pd.concat(frames)
    if "label" in joined.columns:
        joined = joined.set_index("label")
    elif joined.index.name != "label":
        raise NgioValueError(
            "The per-ROI results carry no 'label' column or index, so the "
            "rows cannot be tied to the objects they measure. Add a 'label' "
            "column, or pass a custom `coalesce`."
        )
    return FeatureTable(table_data=joined, reference_label=reference_label)


class FeatureExtractorIterator(AbstractIteratorBuilder[NumpyPipeType, DaskPipeType]):
    """Measure image/label pairs region by region; nothing is written.

    Each unit pairs the image patch with the label patch over the same
    region. `reduce` collects per-region results; `reduce_to_table` joins
    them into a single `FeatureTable` for the caller to store.
    """

    def __init__(
        self,
        input_image: Image,
        input_label: Label,
        *,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (Image): The input image to be used as input for the
                segmentation.
            input_label (Label): The input label with the segmentation masks.
            channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the segmentation.
            axes_order (Sequence[str] | None): Optional axes order for the
                segmentation.
            input_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the input image.
            label_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the output label.
        """
        self._input = input_image
        self._input_label = input_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        # Set iteration parameters
        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._label_transforms = label_transforms

        self._input.require_axes_match(self._input_label)
        self._input.require_rescalable(self._input_label)

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "input_label": self._input_label,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "label_transforms": self._label_transforms,
        }

    def build_numpy_getter(self, roi: Roi) -> FeatureGetter[np.ndarray]:
        data_getter = NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = NumpyGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=roi,
            remove_channel_selection=True,
        )
        return FeatureGetter(data_getter, label_getter)

    def build_dask_getter(self, roi: Roi) -> FeatureGetter[da.Array]:
        data_getter = DaskGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = DaskGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=roi,
            remove_channel_selection=True,
        )
        return FeatureGetter(data_getter, label_getter)

    def build_numpy_setter(self, roi: Roi) -> None:
        return None

    def build_dask_setter(self, roi: Roi) -> None:
        return None

    def finalize(self):
        pass

    def _partials_handler(self):
        """Partials live beside the input label's levels, like a stitch scratch."""
        return self._input_label._group_handler

    def _prepare_distributed(self, n_jobs: int) -> None:
        prepare_partials(self, self._partials_handler(), n_jobs)

    def _validate_job_context(self, n_jobs: int) -> None:
        validate_partials_context(self, self._partials_handler(), n_jobs)

    def reduce_to_table(
        self,
        func: Callable[[np.ndarray, np.ndarray, Roi], FeatureFuncResult],
        *,
        coalesce: Callable[[list[FeatureFuncResult]], Table] | None = None,
        mapper: MapperProtocol | None = None,
    ) -> Table:
        """Measure every ROI and join the results into one table.

        The per-ROI measurement fans out exactly like `reduce` — pass a
        `mapper` to parallelize it — and the join happens once, at the end,
        on the calling thread. Nothing is written: the returned table is
        yours to store, e.g. `container.add_table(name, table)`.

        Args:
            func: `(image, label, roi) -> DataFrame | dict[str, list]` — the
                measurements for one ROI. Rows must carry the object id in a
                `label` column (or index). Under a parallel mapper it runs on
                worker threads or processes and must be safe there.
            coalesce: Joins the per-ROI results into an ngio table. The
                default concatenates them into a `FeatureTable` indexed by
                `label`, referencing the input label.
            mapper: How the per-ROI work is scheduled; `None` is serial.

        Returns:
            The joined table — a `FeatureTable` under the default `coalesce`.
        """
        if self._partition is not None:
            raise NgioValueError(
                "reduce_to_table on a partition slice would coalesce only "
                "this job's share. Use `reduce_to_partial(func)` per job and "
                "`merge_partials()` once, after all jobs."
            )
        results = self.reduce(_UnpackedFeatureFunc(func), mapper=mapper)
        if coalesce is None:
            return _default_feature_coalesce(
                results, reference_label=self._input_label.meta.name
            )
        return coalesce(results)

    def reduce_to_partial(
        self,
        func: Callable[[np.ndarray, np.ndarray, Roi], FeatureFuncResult],
        *,
        mapper: MapperProtocol | None = None,
    ) -> None:
        """Measure this job's share and store the raw records as a partial.

        The per-job step of a distributed run (see `prepare_jobs`): the
        measurements are stored **before** any join, tagged with their global
        ROI index, so `merge_partials` can rebuild the full per-ROI result
        list and run the ONE global coalesce — bit-identical to a serial
        `reduce_to_table`, custom `coalesce` included. Only callable on a
        `for_job` slice; re-running a job overwrites its own partial.

        Frames returned as dicts are normalized to DataFrames and a `label`
        index to a `label` column — a custom `coalesce` at merge time sees
        that normalized form.
        """
        if self._partition is None:
            raise NgioValueError(
                "reduce_to_partial is the per-job step of a distributed run: "
                "restrict the iterator first (`for_job(**args)`), or use "
                "reduce_to_table() for a local run."
            )
        _, job_index, n_jobs = self._partition
        handler = self._partials_handler()
        validate_partials_context(self, handler, n_jobs)

        indices = self.partition_indices or []
        results = self.reduce(_UnpackedFeatureFunc(func), mapper=mapper)
        frames = []
        for index, result in zip(indices, results, strict=True):
            frame = _as_feature_frame(result)
            if INDEX_COLUMN in frame.columns:
                raise NgioValueError(
                    f"The feature function returned a reserved column "
                    f"{INDEX_COLUMN!r}; rename it."
                )
            if not len(frame):
                continue
            if frame.index.name == "label":
                frame = frame.reset_index()
            else:
                frame = frame.reset_index(drop=True)
            frame = frame.copy()
            frame[INDEX_COLUMN] = index
            frames.append(frame)
        payload = pd.concat(frames, ignore_index=True) if frames else None
        write_partial(
            handler,
            job_index,
            payload,
            attrs={
                "job_index": job_index,
                "n_jobs": n_jobs,
                "fingerprint": self._plan_fingerprint(n_jobs),
            },
        )

    def merge_partials(
        self,
        *,
        coalesce: Callable[[list[FeatureFuncResult]], Table] | None = None,
    ) -> Table:
        """Merge a distributed run's partials into the one final table.

        The consolidate step (see `prepare_jobs`): validates that every job
        of the prepared plan produced a matching partial — a half-finished
        run errors instead of returning a silently incomplete table —
        rebuilds the per-ROI result list in global ROI order, and runs the
        single (default or custom) `coalesce`, exactly as a serial
        `reduce_to_table` would. On success the partials group is removed.
        Nothing is registered: the returned table is yours to store with
        `add_table`.
        """
        if self._partition is not None:
            raise NgioValueError(
                "merge_partials belongs to the unrestricted iterator: rebuild "
                "it without `for_job` for the consolidate step."
            )
        handler = self._partials_handler()
        merged = merge_partial_frames(self, handler)

        groups: dict[int, pd.DataFrame] = {}
        if merged is not None:
            for index, group in merged.groupby(INDEX_COLUMN, sort=True):
                groups[int(cast("Any", index))] = group.drop(
                    columns=[INDEX_COLUMN]
                ).reset_index(drop=True)
        results: list[FeatureFuncResult] = [
            groups.get(index, pd.DataFrame()) for index in range(len(self.rois))
        ]
        if coalesce is None:
            table: Table = _default_feature_coalesce(
                results, reference_label=self._input_label.meta.name
            )
        else:
            table = coalesce(results)
        delete_partials_root(handler)
        return table

    def iter_as_numpy(self):  # type: ignore[override]
        """Create an iterator over the pixels of the ROIs."""
        return self._iter(lazy=False, data_mode="numpy", iterator_mode="readonly")

    @deprecated(
        replacement="iter_as_numpy() (or Image.get_as_dask() for a lazy array)",
        removed_in="1.2",
    )
    def iter_as_dask(self):  # type: ignore[override]
        """Create an iterator over the pixels of the ROIs.

        Deprecated: removed in ngio=1.2.
        """
        return self._iter(lazy=False, data_mode="dask", iterator_mode="readonly")
