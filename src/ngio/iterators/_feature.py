from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar, cast

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
    unit_columns_record,
    validate_partials_context,
    write_partial,
)
from ngio.tables import FeatureTable, Table
from ngio.utils import NgioValueError

NumpyPipeType: TypeAlias = tuple[np.ndarray, np.ndarray, Roi]
DaskPipeType: TypeAlias = tuple[da.Array, da.Array, Roi]

T = TypeVar("T", np.ndarray, da.Array)

FeatureFuncResult: TypeAlias = pd.DataFrame | dict[str, list]
"""What a per-ROI feature function may return: a DataFrame, or the cheaper
dict-of-columns (`{"label": [...], "area": [...]}`) — lighter to build in a
worker and to ship across a `ProcessMapper` boundary; normalization turns
either into one DataFrame per ROI before the join."""

#: Provenance columns stamped on every normalized result row: the ROI's
#: global index and its `Roi.get_name()`. Reserved alongside the partials'
#: `INDEX_COLUMN`: a function whose result already carries one is refused.
ROI_INDEX_COLUMN = "roi_index"
ROI_NAME_COLUMN = "roi_name"
_RESERVED_RESULT_COLUMNS = (INDEX_COLUMN, ROI_INDEX_COLUMN, ROI_NAME_COLUMN)


class FeatureGetter(DataGetter[tuple[T, T, Roi]], Generic[T]):
    """Pairs an image getter with a label getter over the same ROI.

    Under a halo both patches and the `roi` this getter reports cover the
    grown region — the ROI always matches the patch extent.
    """

    def __init__(
        self,
        image_getter: DataGetter[T],
        label_getter: DataGetter[T],
    ) -> None:
        self._image_getter = image_getter
        self._label_getter = label_getter
        # Read at most once per `get()` cycle: the properties cache so that
        # `.image`/`.label` followed by `get()` does not pay double IO, and
        # `get()` releases the cache so a getter kept alive in a unit list
        # (e.g. by `reduce`) does not retain its patches after being consumed.
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
        """Return `(image, label, roi)`, releasing the property cache."""
        image, label = self.image, self.label
        self._image_data = None
        self._label_data = None
        return image, label, self.roi

    @property
    def image(self) -> T:
        """The image patch; cached until the next `get()`."""
        if self._image_data is None:
            self._image_data = self._image_getter()
        return self._image_data

    @property
    def label(self) -> T:
        """The label patch; cached until the next `get()`."""
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
        f"columns, got {type(result).__name__}. Declare a custom join "
        "(`with_join`) to handle other shapes."
    )


def _roi_display_name(roi: Roi, index: int) -> str:
    """The ROI's name for the provenance column.

    `Roi.get_name()` falls back to the full repr for an unnamed, unlabeled
    ROI (the whole-image default) — too noisy for a table cell, so those
    fall back to `roi_{index}` instead.
    """
    if roi.name is not None:
        return roi.name
    if roi.label is not None:
        return str(roi.label)
    return f"roi_{index}"


def _normalized_frame(
    result: FeatureFuncResult, *, index: int, roi: Roi
) -> pd.DataFrame:
    """One ROI's result in the normalized form every join sees.

    Dicts become DataFrames; a `label` index becomes a `label` column; a
    zero-row result collapses to the empty, column-less frame; every
    surviving row is stamped with `roi_index` (the ROI's global index) and
    `roi_name` (the ROI's name, `roi_{index}` when it has none).
    """
    frame = _as_feature_frame(result)
    for reserved in _RESERVED_RESULT_COLUMNS:
        if reserved in frame.columns:
            raise NgioValueError(
                f"The feature function returned a reserved column "
                f"{reserved!r}; rename it."
            )
    if not len(frame):
        return pd.DataFrame()
    if frame.index.name == "label":
        frame = frame.reset_index()
    else:
        frame = frame.reset_index(drop=True)
    frame = frame.copy()
    frame[ROI_INDEX_COLUMN] = index
    frame[ROI_NAME_COLUMN] = _roi_display_name(roi, index)
    return frame


class JoinProtocol(Protocol):
    """Joins the normalized per-ROI frames into one ngio table.

    Receives the list described in `measure` — one DataFrame per ROI in
    global ROI order, `label` as a column, every row stamped with
    `roi_index`/`roi_name`, empty ROIs as empty column-less frames — and
    returns the table. Plain callables satisfy this structurally. Runs
    once, wherever the join executes: a serial `measure`, or the
    distributed gather's `finalize`. Must be deterministic, or a
    distributed gather is no longer bit-identical to a serial run.
    """

    def __call__(self, results: list[pd.DataFrame]) -> Table:
        """Return the joined table."""
        ...


@dataclass(frozen=True)
class ConcatJoin:
    """The default join: concatenate into a `FeatureTable` indexed by `label`.

    The rows must carry the object id in a `label` column (or already sit
    on a `label` index). The stamped `roi_index`/`roi_name` columns ride
    into the final table. Duplicate label ids (a haloed or overlapping
    tiling measures border objects more than once) are kept as-is —
    deduplicating is a custom join's job. The zero-object table keeps its
    label-only schema.
    """

    reference_label: str | None = None

    def __call__(self, results: list[pd.DataFrame]) -> FeatureTable:
        """Concatenate the normalized frames into one `FeatureTable`."""
        if not results:
            raise NgioValueError(
                "No per-ROI results to build a table from: the iterator has "
                "no ROIs. Check the ROI table or the tiling that produced it."
            )
        frames = [_as_feature_frame(result) for result in results]
        # ROIs with no objects contribute empty frames; concatenating those
        # would silently upcast the surviving columns (an empty `label` is
        # float64, and a float index fails the integer-index validation).
        frames = [frame for frame in frames if len(frame)]
        if not frames:
            # Zero objects is a legitimate outcome, not an error — the same
            # contract as `detect`. No rows also means no feature columns.
            empty = pd.DataFrame({"label": pd.Series([], dtype="int64")})
            return FeatureTable(
                table_data=empty.set_index("label"),
                reference_label=self.reference_label,
            )
        joined = pd.concat(frames)
        if "label" in joined.columns:
            joined = joined.set_index("label")
        elif joined.index.name != "label":
            raise NgioValueError(
                "The per-ROI results carry no 'label' column or index, so "
                "the rows cannot be tied to the objects they measure. Add a "
                "'label' column, or declare a custom join with "
                "`with_join(...)`."
            )
        return FeatureTable(table_data=joined, reference_label=self.reference_label)


class FeatureExtractorIterator(
    AbstractIteratorBuilder[NumpyPipeType, DaskPipeType, Table]
):
    """Measure image/label pairs region by region; nothing is written.

    Each unit pairs the image patch with the label patch over the same
    region. `reduce` collects per-region results; `measure` joins them into
    a single `FeatureTable` for the caller to store. Distributed, a
    `for_job` slice's `measure` banks a partial and `finalize()` runs the
    one global join (declared with `with_join`, `ConcatJoin` otherwise).
    `with_halo(...)` reads context around each region; the duplicate rows
    it produces for border objects are reconciled in your declared join
    via the stamped `roi_index`/`roi_name` columns.
    """

    # The halo is a pure read margin here too: there is no write to crop it
    # from. Unlike detection, this iterator does not reconcile the overlap
    # itself — a border object measured by two grown regions yields
    # duplicate label rows, and the stamped `roi_index`/`roi_name` columns
    # are the handle the declared join uses to reconcile them.
    _allow_readonly_halo = True
    # The declared join; `None` runs `ConcatJoin` against the input label.
    _join: JoinProtocol | None = None
    _chain_state_attrs: tuple[str, ...] = ("_join",)

    def with_join(self, join: JoinProtocol) -> Self:
        """Declare the join that turns the per-ROI results into the table.

        Runs once, wherever the join executes — a serial `measure`, or the
        distributed gather's `finalize`. On a `for_job` slice the
        declaration is inert: the slice banks the pre-join records
        regardless, so declaring it there is harmless. Like the
        measurement function, the join is not part of the plan
        fingerprint: declare the identical one on the gather iterator.
        The default (no declaration) is `ConcatJoin` referencing the
        input label.

        Args:
            join: Anything satisfying `JoinProtocol` — a plain callable
                taking the normalized frame list, or a configured class
                like `ConcatJoin(reference_label=...)`.
        """
        if not callable(join):
            raise NgioValueError(
                f"The join must be callable, got {type(join).__name__}."
            )
        new_instance = self._new_from_rois(self.rois)
        new_instance._join = join
        return new_instance

    def _resolved_join(self) -> JoinProtocol:
        """The declared join, or the default concat against the input label."""
        if self._join is not None:
            return self._join
        return ConcatJoin(reference_label=self._input_label.meta.name)

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
        """Measure `input_image`/`input_label` pairs; nothing is written.

        Args:
            input_image: The image to measure.
            input_label: The label whose objects tie the measurements together.
            channel_selection: Restrict the image reads to these channels.
            axes_order: Axes order of the patches handed to the function.
            input_transforms: Transforms applied to each image patch.
            label_transforms: Transforms applied to each label patch.
        """
        self._input = input_image
        self._input_label = input_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._label_transforms = label_transforms

        self._input.require_axes_match(self._input_label)
        self._input.require_rescalable(self._input_label)

    def _get_init_kwargs(self) -> dict:
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
        # Both getters take the same halo-grown world-space ROI; each
        # converts at its own image's pixel size, so a coarser label
        # rescales the margin correctly on its own.
        read_roi = self._read_roi(roi)
        data_getter = NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=read_roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = NumpyGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=read_roi,
            remove_channel_selection=True,
        )
        return FeatureGetter(data_getter, label_getter)

    def build_dask_getter(self, roi: Roi) -> FeatureGetter[da.Array]:
        read_roi = self._read_roi(roi)
        data_getter = DaskGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=read_roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = DaskGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=read_roi,
            remove_channel_selection=True,
        )
        return FeatureGetter(data_getter, label_getter)

    def finalize(self) -> Table:
        """Merge a distributed run's partials into the one final table.

        The gather step (see `prepare_jobs`): validates that every job of
        the prepared plan produced a matching partial — a half-finished run
        errors instead of returning a silently incomplete table — rebuilds
        the per-ROI result list in global ROI order, and runs the single
        declared join (`with_join`, `ConcatJoin` otherwise) exactly once.
        The final table matches a serial `measure` row for row; the result
        list is the normalized form described in `measure`,
        `roi_index`/`roi_name` included. On success the partials group is
        removed. Nothing is registered: the returned table is yours to
        store with `add_table`.

        Raises on a `for_job` slice (the gather is global) and when no
        partials exist (nothing was prepared or banked).
        """
        self._require_unrestricted_finalize()
        handler = self._partials_handler()
        merged = merge_partial_frames(self, handler, job_verb="measure")

        groups: dict[int, pd.DataFrame] = {}
        if merged.frame is not None:
            for index, group in merged.frame.groupby(INDEX_COLUMN, sort=True):
                frame = group.drop(columns=[INDEX_COLUMN]).reset_index(drop=True)
                record = merged.columns.get(int(cast("Any", index)))
                if record is not None:
                    # The concat unioned columns across ROIs (NaN-filled,
                    # ints upcast); restore this ROI's own schema exactly.
                    columns, dtypes = record
                    frame = frame[columns].astype(dtypes)
                else:
                    # Partial from an older ngio with no schema record: pin
                    # the provenance columns back to the end, where a serial
                    # normalization puts them.
                    measured = [
                        column
                        for column in frame.columns
                        if column not in (ROI_INDEX_COLUMN, ROI_NAME_COLUMN)
                    ]
                    frame = frame[[*measured, ROI_INDEX_COLUMN, ROI_NAME_COLUMN]]
                groups[int(cast("Any", index))] = frame
        results: list[pd.DataFrame] = [
            groups.get(index, pd.DataFrame()) for index in range(len(self.rois))
        ]
        table = self._resolved_join()(results)
        delete_partials_root(handler)
        return table

    def _partials_handler(self):
        """Partials live beside the input label's levels, like a stitch scratch."""
        return self._input_label._group_handler

    def _prepare_distributed(self, n_jobs: int) -> None:
        prepare_partials(self, self._partials_handler(), n_jobs)

    def _validate_job_context(self, n_jobs: int) -> None:
        validate_partials_context(self, self._partials_handler(), n_jobs)

    def measure(
        self,
        func: Callable[[np.ndarray, np.ndarray, Roi], FeatureFuncResult],
        *,
        mapper: MapperProtocol | None = None,
    ) -> Table | None:
        """Measure every ROI and join the results into one table.

        The per-ROI measurement fans out exactly like `reduce` — pass a
        `mapper` to parallelize it — and the declared join (`with_join`,
        `ConcatJoin` otherwise) runs once, at the end, on the calling
        thread. Nothing is written: the returned table is yours to store,
        e.g. `container.add_table(name, table)`.

        The join (declared or default, serial or distributed) always sees
        the *normalized* result list, not the function's raw returns: one
        DataFrame per ROI with `label` as a column (a `label` index is
        reset), every row stamped with `roi_index` (the ROI's global index)
        and `roi_name` (the ROI's name — `roi_{index}` when it has none, as
        seen by the job that measured it), and a ROI whose function
        returned no rows as an empty, column-less DataFrame. The three
        names `_ngio_index`, `roi_index` and `roi_name` are reserved: a
        function whose result carries one is refused.

        With `with_halo(...)` each region is read grown — both patches and
        the `roi` argument cover the grown region — so a border object is
        measured by every region that sees it. The default join keeps the
        resulting duplicate label rows as-is; reconcile them in a declared
        join via the `roi_index`/`roi_name` columns. `reduce` and `iter`
        read the grown regions too.

        On a `for_job` slice it measures only this job's share and banks the
        normalized records as a partial, returning `None` — stored
        **before** any join, so the gather (`finalize()`, once, after all
        jobs) can rebuild the full per-ROI result list and run the ONE
        global join. Re-running a job overwrites its own partial.

        Args:
            func: `(image, label, roi) -> DataFrame | dict[str, list]` — the
                measurements for one ROI. Rows must carry the object id in a
                `label` column (or index). Under a parallel mapper it runs on
                worker threads or processes and must be safe there.
            mapper: How the per-ROI work is scheduled; `None` is serial.

        Returns:
            The joined table — a `FeatureTable` under the default join. A
            run that finds zero objects returns an empty table, as `detect`
            does. On a `for_job` slice: `None` (the partial is banked).
        """
        if self._partition is not None:
            self._bank_partial(func, mapper=mapper)
            return None
        results = self.reduce(_UnpackedFeatureFunc(func), mapper=mapper)
        normalized = [
            _normalized_frame(result, index=index, roi=roi)
            for index, (roi, result) in enumerate(zip(self.rois, results, strict=True))
        ]
        return self._resolved_join()(normalized)

    def _bank_partial(
        self,
        func: Callable[[np.ndarray, np.ndarray, Roi], FeatureFuncResult],
        *,
        mapper: MapperProtocol | None = None,
    ) -> None:
        """Measure this job's share and store the raw records as a partial."""
        assert self._partition is not None
        _, job_index, n_jobs = self._partition
        handler = self._partials_handler()
        validate_partials_context(self, handler, n_jobs)

        indices = self.partition_indices or []
        results = self.reduce(_UnpackedFeatureFunc(func), mapper=mapper)
        frames = []
        column_records = []
        for index, result in zip(indices, results, strict=True):
            frame = _normalized_frame(result, index=index, roi=self.rois[index])
            if not len(frame):
                continue
            # Recorded before the merge key: the concat below unions columns
            # across ROIs, and the gather restores each ROI's own schema.
            column_records.append(unit_columns_record(index, frame))
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
                "columns": column_records,
            },
        )

    # `iter`/`iter_as_numpy` come from the read-only base and yield
    # `(image, label, roi)` payloads; there is nothing to write.
