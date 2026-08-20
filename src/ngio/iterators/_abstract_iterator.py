import hashlib
import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from types import MappingProxyType
from typing import Generic, Literal, Self, TypedDict, TypeVar, overload

from ngio.common import Roi
from ngio.common._pyramid import RegionsLike
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._ops_slices_utils import (
    _pairs_stream,
    check_if_regions_overlap,
    chunk_rects_intersect,
)
from ngio.iterators._halo import HaloCroppingSetter
from ngio.iterators._mappers import (
    BasicMapper,
    IterUnit,
    MapperProtocol,
    compute_write_footprint,
    write_conflict_components,
)
from ngio.iterators._rois_utils import (
    TailPolicy,
    by_blocks,
    by_grid,
    by_storage_units,
    by_yx,
    by_zyx,
    halo_roi,
    rois_product,
)
from ngio.iterators._split import partition_components
from ngio.tables import GenericRoiTable
from ngio.utils import (
    NgioDeprecationWarning,
    NgioFutureWarning,
    NgioValueError,
    deprecated,
)

NumpyPipeType = TypeVar("NumpyPipeType")
DaskPipeType = TypeVar("DaskPipeType")
R = TypeVar("R")
T = TypeVar("T")


def _is_same_zarr_array(left, right) -> bool:
    """Whether two zarr array handles point at the same stored array."""
    if left is right:
        return True
    try:
        return left.store == right.store and left.path == right.path
    except (AttributeError, TypeError):
        return False


class JobArgs(TypedDict):
    """One entry of `prepare_jobs`'s parallelization list.

    A plain dict at runtime — JSON-serializable for schedulers that ship
    task arguments as JSON (Fractal's parallelization list), and it splats
    straight into the selector: `iterator.for_job(**args)`.
    """

    job_index: int
    n_jobs: int


class AbstractIteratorBuilder(ABC, Generic[NumpyPipeType, DaskPipeType]):
    """Base class for building iterators over ROIs.

    Constructor naming convention across the concrete iterators: a bare
    parameter (`channel_selection`, `input_transforms`) refers to the *input*
    image; output-side parameters are always prefixed
    (`output_channel_selection`, `output_transforms`), and parameters
    targeting another object name it (`label_transforms`).
    """

    _rois: list[Roi]
    _ref_image: AbstractImage
    # Extra pixels read around each ROI and dropped before writing; see
    # `with_halo`. Empty means the read and the write cover the same region.
    # The default is immutable and shared: instances assign a fresh dict, and
    # a mutable class-level `{}` would be one shared across every iterator.
    _halo: Mapping[str, int] = MappingProxyType({})
    # A read-only subclass that opts in declares the halo a pure read margin:
    # there is no write to crop it from, and the subclass owns reconciling the
    # overlapping context (the detection iterator's NMS). Off by default so a
    # read-only iterator does not silently measure grown regions.
    _allow_readonly_halo: bool = False
    # Set by `for_job`: (selected unit indices, job_index, n_jobs).
    # `None` means unrestricted. Deliberately not carried through
    # `_new_from_rois`: the indices are positions in the current ROI list, so
    # a partition slice refuses further reshaping instead of silently
    # invalidating them.
    _partition: tuple[frozenset[int], int, int] | None = None

    def __repr__(self) -> str:
        halo = f", halo={self._halo}" if self._halo else ""
        partition = ""
        if self._partition is not None:
            _, job_index, n_jobs = self._partition
            partition = f", partition={job_index}/{n_jobs}"
        return f"{self.__class__.__name__}(regions={len(self._rois)}{halo}{partition})"

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

    @property
    def output_image(self) -> AbstractImage | None:
        """The image this iterator writes to, or `None` for a read-only iterator."""
        return None

    @classmethod
    def _require_complete_init_kwargs(cls, init_kwargs: dict) -> None:
        """Refuse a `get_init_kwargs` that silently forgets constructor state.

        Every reshaping call rebuilds the iterator from this dict; a
        constructor parameter missing from it is state that vanishes on the
        first `.by_grid()`/`.by_chunks()`/`.with_halo()` with no error. Checked
        once per class, then cached.
        """
        checked = cls.__dict__.get("_init_kwargs_checked", False)
        if checked:
            return
        parameters = inspect.signature(cls.__init__).parameters
        missing = [
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            and name not in init_kwargs
        ]
        if missing:
            raise NgioValueError(
                f"{cls.__name__}.get_init_kwargs() omits the constructor "
                f"parameter(s) {missing}: the iterator would silently lose "
                "that state on the first by_grid()/by_chunks()/with_halo() "
                "call. Add them to get_init_kwargs()."
            )
        cls._init_kwargs_checked = True

    def _new_from_rois(self, rois: list[Roi]) -> Self:
        """Create a new instance of the iterator with a different set of ROIs."""
        if self._partition is not None:
            raise NgioValueError(
                "This iterator is restricted to one partition "
                "(`for_job`), and its selection is a set of positions "
                "in the current ROI list — reshaping it (`by_*`, `product`, "
                "`with_halo`) would silently invalidate the selection. Apply "
                "`for_job` last, after every reshaping call."
            )
        init_kwargs = self.get_init_kwargs()
        self._require_complete_init_kwargs(init_kwargs)
        new_instance = self.__class__(**init_kwargs)
        new_instance._set_rois(rois)
        # Carried here rather than through every concrete `get_init_kwargs`:
        # the halo is orthogonal to how an iterator was built.
        new_instance._halo = self._halo
        return new_instance

    def with_halo(self, x: int = 0, y: int = 0, z: int = 0, t: int = 0) -> Self:
        """Read a margin of context around each ROI, but write only the ROI.

        The function is handed the grown region and must return it grown too;
        the extra border is cropped off before the write. This is how you get
        seamless tiles — each tile sees past its own edge, so a smoothing or a
        segmentation has the context it needs — and how you "trim" a tile,
        which is the same operation seen from the other side.

        On a read-only iterator that opts in (the object-detection iterator),
        the halo is a pure read margin: there is no write to crop it from, and
        the iterator itself reconciles the overlapping context (NMS resolves
        the duplicate detections the margin produces). Other read-only
        iterators refuse, so a `reduce` never silently measures grown regions.

        The ROIs themselves do not move, so the write footprints are unchanged
        and a haloed iterator parallelizes exactly as far as it did without
        one. That is the point of doing this on the read side: overlapping
        *writes* would have to be serialized, overlapping reads cost nothing.

        Margins are in pixels of the reference image and are clipped at its
        borders, so an edge tile simply grows on the sides where there is room.

        Note:
            Because the margins are counted in reference-image pixels, they do
            not survive a rescale: a shape-changing transform on the data path
            (a `ZoomTransform`, say) leaves the patch on a different grid from
            the margins, and the crop refuses rather than guessing. To work at
            a lower resolution, iterate on a coarser pyramid level instead. A
            `MaskTransform` is unaffected — its zoom rescales the *label* it
            reads, never the data array — and composes with a halo normally.

        Args:
            x: Pixels added on each side along x.
            y: Pixels added on each side along y.
            z: Pixels added on each side along z.
            t: Frames added on each side along t.

        Returns:
            A new iterator reading with the halo.

        Raises:
            NgioValueError: On a read-only iterator, which has no write region
                for the halo to be defined against, or for a negative margin.

        Example:
            ```python
            # 8 px of context per side, disjoint writes, still parallel.
            it = iterator.by_write_units().with_halo(x=8, y=8)
            it.map(smooth, mapper=ThreadedMapper("auto"))
            ```
        """
        halo = {"x": x, "y": y, "z": z, "t": t}
        for axis_name, margin in halo.items():
            if margin < 0:
                raise NgioValueError(
                    f"Halo along '{axis_name}' must be >= 0, got {margin}."
                )
        if self.output_image is None and not self._allow_readonly_halo:
            name = self.__class__.__name__
            raise NgioValueError(
                f"{name} is read-only, so there is no written region for a "
                "halo to surround. Widen the ROIs themselves instead (e.g. "
                "`by_grid(...)` with a stride smaller than the size)."
            )
        output = self.output_image
        if output is not None and _is_same_zarr_array(
            self._ref_image.zarr_array, output.zarr_array
        ):
            raise NgioValueError(
                "In-place iteration cannot take a halo: each tile's grown "
                "read covers pixels that neighbouring tiles write, so the "
                "result depends on execution order. Write to a different "
                "image or label, or drop the halo."
            )
        new_instance = self._new_from_rois(self.rois)
        new_instance._halo = {ax: m for ax, m in halo.items() if m > 0}
        return new_instance

    @property
    def halo(self) -> dict[str, int]:
        """The per-axis read margin, empty when there is none."""
        return dict(self._halo)

    def _read_roi(self, roi: Roi) -> Roi:
        """The region a getter reads for `roi`: grown by the halo, if any."""
        if not self._halo:
            return roi
        read_roi, _ = halo_roi(roi, self._ref_image, self._halo)
        return read_roi

    def _wrap_setter(
        self, setter: DataSetterProtocol[T], roi: Roi
    ) -> DataSetterProtocol[T]:
        """Crop the halo off patches heading into `setter`, if there is one."""
        if not self._halo:
            return setter
        _, margins = halo_roi(roi, self._ref_image, self._halo)
        if not margins:
            return setter
        return HaloCroppingSetter(setter, margins)

    def by_grid(
        self,
        *,
        size_x: int | None = None,
        size_y: int | None = None,
        size_z: int | None = None,
        size_t: int | None = None,
        stride_x: int | None = None,
        stride_y: int | None = None,
        stride_z: int | None = None,
        stride_t: int | None = None,
        tail: TailPolicy = "clip",
        base_name: str = "",
    ) -> Self:
        """Tile the current ROIs with a regular grid.

        Args:
            size_x: Tile size along x; `None` (default) means the full extent.
            size_y: Tile size along y; `None` means the full extent.
            size_z: Tile size along z; `None` means the full extent.
            size_t: Tile size along t; `None` means the full extent.
            stride_x: Step between tiles along x; `None` means `size_x`
                (adjacent tiles). A smaller stride produces overlapping tiles.
            stride_y: As `stride_x`, along y.
            stride_z: As `stride_x`, along z.
            stride_t: As `stride_x`, along t.
            tail: What happens where the grid does not divide the axis.
                `"clip"` (default) shrinks the last tile to the border;
                `"balance"` re-splits the last full tile and the tail into two
                near-equal tiles, so a thin overhang never yields a thin tile
                (needs adjacent tiles, `stride == size`); `"shift"` moves the
                last tile back to stay full-size — it then overlaps its
                neighbour, so a writing iterator needs a merge or serial
                execution; `"drop"` discards the tail and leaves that border
                uncovered.
            base_name: Prefix for the generated tile names.

        Returns:
            A new iterator over the tiled ROIs.
        """
        rois = by_grid(
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
            tail=tail,
            base_name=base_name,
        )
        return self._new_from_rois(rois)

    def by_blocks(
        self,
        *,
        num_x: int = 1,
        num_y: int = 1,
        num_z: int = 1,
        num_t: int = 1,
        base_name: str = "",
    ) -> Self:
        """Divide each axis into a fixed number of near-equal blocks.

        The complement of `by_grid`: you say how many tiles, not how big.
        Block lengths along an axis differ by at most one pixel, so there is
        no tail to police — the partition is balanced by construction.

        Args:
            num_x: Number of blocks along x (default 1: no split).
            num_y: Number of blocks along y.
            num_z: Number of blocks along z.
            num_t: Number of blocks along t.
            base_name: Prefix for the generated tile names.

        Returns:
            A new iterator over the blocks.
        """
        rois = by_blocks(
            rois=self.rois,
            ref_image=self.ref_image,
            num_x=num_x,
            num_y=num_y,
            num_z=num_z,
            num_t=num_t,
            base_name=base_name,
        )
        return self._new_from_rois(rois)

    def by_yx(self) -> Self:
        """Split each ROI into one 2D plane per remaining coordinate.

        A broadcasting call, not a tiling: every ROI becomes one region per
        `(t, z, c)` combination, each covering the full y/x extent — the
        shape for "run this 2D function on every plane".
        """
        rois = by_yx(self.rois, self.ref_image)
        return self._new_from_rois(rois)

    def by_zyx(self, strict: bool = True) -> Self:
        """Split each ROI into one 3D volume per remaining coordinate.

        A broadcasting call, not a tiling: every ROI becomes one region per
        `(t, c)` combination, each covering the full z/y/x extent.

        Args:
            strict: Require a real z axis (present and larger than 1);
                with `False`, images without one fall back to 2D planes.
        """
        rois = by_zyx(self.rois, self.ref_image, strict=strict)
        return self._new_from_rois(rois)

    def by_chunks(
        self,
        *,
        overlap_x: int = 0,
        overlap_y: int = 0,
        overlap_z: int = 0,
        overlap_t: int = 0,
    ) -> Self:
        """Tile the ROIs by the input image's chunk grid.

        Reads are chunk-granular, so this is the natural tiling for
        read-heavy work. For collision-free parallel *writes* use
        `by_write_units`, which tiles by the output's write granularity.

        Args:
            overlap_x: Overlap between adjacent tiles along x, in pixels.
            overlap_y: Overlap along y.
            overlap_z: Overlap along z.
            overlap_t: Overlap along t.

        Returns:
            A new iterator with tiled ROIs.
        """
        rois = by_storage_units(
            self.rois,
            self.ref_image,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            overlap_z=overlap_z,
            overlap_t=overlap_t,
            grid_image=None,
        )
        return self._new_from_rois(rois)

    def by_write_units(
        self,
        *,
        overlap_x: int = 0,
        overlap_y: int = 0,
        overlap_z: int = 0,
        overlap_t: int = 0,
    ) -> Self:
        """Tile the ROIs by the output image's write granularity.

        The write unit is the shard shape when the output is sharded (writes
        are atomic per shard object), the chunk shape otherwise. With no
        overlap the resulting ROIs pass `check_if_write_units_overlap` by
        construction, so a parallel `map` runs as a single fully-parallel
        wave — a throughput optimization; conflicting tilings are still safe,
        just scheduled in more waves. Falls back to the input chunk grid when
        the iterator is read-only.

        Args:
            overlap_x: Overlap between adjacent tiles along x, in pixels.
            overlap_y: Overlap along y.
            overlap_z: Overlap along z.
            overlap_t: Overlap along t.

        Returns:
            A new iterator with tiled ROIs.
        """
        rois = by_storage_units(
            self.rois,
            self.ref_image,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            overlap_z=overlap_z,
            overlap_t=overlap_t,
            grid_image=self.output_image,
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
    def finalize(self) -> None:
        """Run once after every unit completes.

        Writing iterators resolve any pending reconciliation (the stitch) and
        consolidate the output pyramid here; read-only iterators do nothing.
        """
        raise NotImplementedError

    def _touched_write_regions(self) -> RegionsLike | None:
        """The on-disk regions this iterator's setters write, or `None`.

        Derived from the ROI list on the calling side rather than accumulated
        by the setters: setters run inside `ProcessMapper` children, and
        nothing accumulated there survives the process boundary. The ROI list
        is the parent's source of truth for what was written -- which is why
        writing iterators can hand this to `consolidate(regions=...)` in
        `finalize` and rebuild only the pyramid regions they touched.

        Halo and stitch wrappers forward the core write's `slicing_ops`, so
        haloed reads and banked stitch bands do not inflate the regions.
        Building a setter per ROI is pure metadata -- no pixel IO.
        """
        regions = []
        for roi in self.rois:
            setter = self.build_numpy_setter(roi)
            if setter is None:
                return None
            regions.append(setter.slicing_ops.normalized_slicing_tuple)
        return regions or None

    def _numpy_getters_generator(self) -> Generator[DataGetterProtocol[NumpyPipeType]]:
        """Return a list of numpy getter functions for all (selected) ROIs."""
        yield from (self.build_numpy_getter(roi) for _, roi in self._selected_rois())

    def _dask_getters_generator(self) -> Generator[DataGetterProtocol[DaskPipeType]]:
        """Return a list of dask getter functions for all (selected) ROIs."""
        yield from (self.build_dask_getter(roi) for _, roi in self._selected_rois())

    def _numpy_setters_generator(
        self,
    ) -> Generator[DataSetterProtocol[NumpyPipeType] | None]:
        """Return a list of numpy setter functions for all (selected) ROIs."""
        yield from (self.build_numpy_setter(roi) for _, roi in self._selected_rois())

    def _dask_setters_generator(
        self,
    ) -> Generator[DataSetterProtocol[DaskPipeType] | None]:
        """Return a list of dask setter functions for all (selected) ROIs."""
        yield from (self.build_dask_setter(roi) for _, roi in self._selected_rois())

    def _selected_rois(self) -> Generator[tuple[int, Roi]]:
        """`(index, roi)` pairs, filtered to the selected partition if any.

        The index is always the position in the *full* ROI list, whether or
        not a partition restricts which ROIs come out — `IterUnit.index`
        stays globally consistent across every job of a split run.
        """
        selection = None if self._partition is None else self._partition[0]
        for index, roi in enumerate(self.rois):
            if selection is None or index in selection:
                yield index, roi

    def _numpy_units_generator(
        self, with_setters: bool = True
    ) -> Generator[IterUnit[NumpyPipeType]]:
        """Yield one numpy `IterUnit` per (selected) ROI, in ROI order."""
        for index, roi in self._selected_rois():
            yield IterUnit(
                index=index,
                roi=roi,
                getter=self.build_numpy_getter(roi),
                setter=self.build_numpy_setter(roi) if with_setters else None,
            )

    def _dask_units_generator(
        self, with_setters: bool = True
    ) -> Generator[IterUnit[DaskPipeType]]:
        """Yield one dask `IterUnit` per (selected) ROI, in ROI order."""
        for index, roi in self._selected_rois():
            yield IterUnit(
                index=index,
                roi=roi,
                getter=self.build_dask_getter(roi),
                setter=self.build_dask_setter(roi) if with_setters else None,
            )

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
        if self._partition is None:
            self.finalize()

    def _iter(
        self,
        lazy: bool = False,
        data_mode: Literal["numpy", "dask"] = "dask",
        iterator_mode: Literal["readwrite", "readonly"] = "readwrite",
    ) -> Generator:
        """Create an iterator over the pixels of the ROIs (no deprecation warnings)."""
        if data_mode == "numpy":
            getters = self._numpy_getters_generator()
            setters = self._numpy_setters_generator()
        elif data_mode == "dask":
            getters = self._dask_getters_generator()
            setters = self._dask_setters_generator()
        else:
            raise NgioValueError(f"Invalid mode: {data_mode}")

        if iterator_mode == "readonly":
            self._require_no_halo_on_readonly_verbs()
            if lazy:
                return getters
            else:
                return (getter() for getter in getters)
        if lazy:
            return self._read_and_write_generator(getters, setters)
        else:
            gen = self._read_and_write_generator(getters, setters)
            return ((getter(), setter) for getter, setter in gen)

    def _iter_batched_readonly(self, batch_size: int) -> Generator[list, None, None]:
        getters = self._iter(lazy=True, data_mode="numpy", iterator_mode="readonly")
        batch: list = []
        for getter in getters:
            batch.append(getter)
            if len(batch) == batch_size:
                yield [getter() for getter in batch]
                batch = []
        if batch:
            yield [getter() for getter in batch]

    def _require_valid_batch_size(self, batch_size: int) -> None:
        if batch_size < 1:
            raise NgioValueError(f"batch_size must be >= 1, got {batch_size}.")

    def iter_batched(
        self, batch_size: int = 8
    ) -> Generator[
        tuple[list[NumpyPipeType], list[DataSetterProtocol[NumpyPipeType]]],
        None,
        None,
    ]:
        """Iterate the ROIs in batches: `(patches, writers)` per batch.

        Two aligned lists of up to `batch_size` items — stack the patches
        yourself (raggedness is yours to handle, unlike `BatchedMapper`'s
        automatic padding), run the model once, and hand each result to its
        writer. Batches follow ROI order — unlike the mappers' canonical
        wave order, so on ROIs with overlapping writes the manual loop is
        deterministic but not bit-identical to `map`. The run finalizes when
        the generator is fully drained, exactly like `iter` — on a stitching
        iterator that finalize is the resolve, so drain the loop to
        completion.

        Args:
            batch_size: Maximum items per batch; the last batch may be
                smaller.

        Example:
            ```python
            for patches, writers in iterator.iter_batched(8):
                outs = model(np.stack(patches))
                for writer, out in zip(writers, outs):
                    writer(out)
            ```
        """
        self._require_valid_batch_size(batch_size)
        getters = self._numpy_getters_generator()
        setters = self._numpy_setters_generator()
        batch: list = []
        for getter, setter in zip(getters, setters, strict=True):
            if setter is None:
                name = self.__class__.__name__
                raise NgioValueError(f"Iterator is read-only: {name}")
            batch.append((getter, setter))
            if len(batch) == batch_size:
                yield [g() for g, _ in batch], [s for _, s in batch]
                batch = []
        if batch:
            yield [g() for g, _ in batch], [s for _, s in batch]
        # After the final batch, like `iter`: the consumer has had its chance
        # to call the writers by the time it requests past the last yield.
        if self._partition is None:
            self.finalize()

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
        data_mode: Literal["numpy", "dask"] | None = None,
        iterator_mode: Literal["readwrite", "readonly"] = "readwrite",
    ) -> Generator:
        """Create an iterator over the pixels of the ROIs.

        A writing loop finalizes only when fully drained — on a stitching
        iterator that finalize is the resolve, so abandoning the loop
        mid-way leaves block-offset ids on disk (plus the transient scratch)
        until a re-run or a later `finalize()`.

        Note:
            The dask data mode is deprecated and will be removed in ngio=1.2;
            from then on `iter()` yields numpy arrays.
        """
        if data_mode is None:
            warnings.warn(
                "iter() currently defaults to data_mode='dask'; in ngio=1.2 it "
                "will yield numpy arrays and the dask mode will be removed. "
                "Pass data_mode='numpy' (or use iter_as_numpy()) to adopt the "
                "future behaviour now, or data_mode='dask' to keep the current "
                "behaviour and silence this warning.",
                NgioFutureWarning,
                stacklevel=2,
            )
            data_mode = "dask"
        elif data_mode == "dask":
            warnings.warn(
                "data_mode='dask' on iter() is deprecated and will be removed "
                "in ngio=1.2; the numpy path is the only iterator backend going "
                "forward. Use data_mode='numpy', or Image.get_as_dask() for "
                "lazy whole-region access.",
                NgioDeprecationWarning,
                stacklevel=2,
            )
        return self._iter(lazy=lazy, data_mode=data_mode, iterator_mode=iterator_mode)

    def iter_as_numpy(
        self,
    ):
        """Create an iterator over the pixels of the ROIs."""
        return self._iter(lazy=False, data_mode="numpy", iterator_mode="readwrite")

    @deprecated(
        replacement="iter_as_numpy() (or Image.get_as_dask() for a lazy array)",
        removed_in="1.2",
    )
    def iter_as_dask(
        self,
    ):
        """Create an iterator over the pixels of the ROIs.

        Deprecated: removed in ngio=1.2.
        """
        return self._iter(lazy=False, data_mode="dask", iterator_mode="readwrite")

    def _require_writable_units(
        self, units: list[IterUnit[NumpyPipeType]] | list[IterUnit[DaskPipeType]]
    ) -> None:
        """Raise if every unit is read-only: there is nothing for `map` to write."""
        # `output_image` first, so a read-only iterator with an *empty* ROI
        # list still refuses instead of silently succeeding.
        if self.output_image is None or (
            units and all(unit.setter is None for unit in units)
        ):
            name = self.__class__.__name__
            raise NgioValueError(
                f"{name} is read-only: map has nothing to write back. "
                "Use reduce (or iter) to compute without writing."
            )

    def _require_no_halo_on_readonly_verbs(self) -> None:
        """Refuse `reduce`/readonly `iter` on a haloed *writing* iterator.

        The halo is defined against the write: read grown, crop, write the
        core. A read-only verb has no write to crop, so it would silently
        hand `func` the grown regions. (Read-only iterators that opt into a
        halo — object detection — reconcile the overlap themselves.)
        """
        if self._halo and self.output_image is not None:
            raise NgioValueError(
                "This iterator writes, and its halo belongs to the write "
                "path (read grown, write the core). A read-only reduce/iter "
                "would silently measure the grown regions instead. Rebuild "
                "the iterator without `with_halo(...)` to measure."
            )

    def _require_splittable(self) -> None:
        """Hook refusing job splitting where it could not be correct.

        The base allows it everywhere: writers distribute their writes, and
        the read-only iterators distribute via their partial verbs
        (`measure_to_partial` / `detect_to_partial`) plus a `merge_partials`
        gather. Subclasses with an unsupported configuration raise here.
        """

    @property
    def partition_indices(self) -> list[int] | None:
        """The unit indices this iterator is restricted to, or `None`.

        `None` on an unrestricted iterator; on a `for_job` slice,
        the sorted positions (in the full ROI list) of the units it runs.
        The whole layout of a split is
        `[it.for_job(i, n).partition_indices for i in range(n)]` —
        the pre-flight check before submitting a distributed run: one fat
        list plus empties means the output chunking, not the cluster, is
        the constraint.
        """
        if self._partition is None:
            return None
        return sorted(self._partition[0])

    def for_job(self, job_index: int, n_jobs: int) -> Self:
        """Restrict the iterator to one of `n_jobs` independent partitions.

        The distributed counterpart of a parallel mapper: partition the
        units into `n_jobs` embarrassingly independent jobs and keep job
        `job_index`'s share. No write unit (a chunk, or a shard when the
        output is sharded) is ever shared between two partitions — units
        whose write footprints conflict travel together — so the jobs need
        no locks, no barriers, and no channel to one another: safe to run
        as separate SLURM array tasks on a shared filesystem, in any order
        and any overlap in time. The partition is a pure function of the
        iterator's ROIs and the output geometry, so every process that
        builds the same iterator derives the same partition on its own.

        The returned iterator executes as usual (`map(func, mapper=...)`),
        with one difference: it does **not** finalize — the pyramid resolve
        is the one global step and belongs to the unrestricted iterator,
        once after all jobs (a SLURM job dependency is the barrier). The
        typical array task, and its dependent gather task:

        ```python
        n_jobs = 4
        job_index = 0  # e.g. int($SLURM_ARRAY_TASK_ID)

        iterator = SegmentationIterator(image, label, ...).by_chunks()
        iterator.for_job(job_index, n_jobs=n_jobs).map(func)

        # gather task, after all jobs:
        SegmentationIterator(image, label, ...).by_chunks().finalize()
        ```

        Every job must build an identical iterator — same construction,
        same tiling calls, same `n_jobs`; construction is metadata-only, so
        this is cheap. `for_job` comes *last* in the builder chain
        (reshaping a slice would invalidate its selection and refuses), and
        in-memory stores cannot work across separate processes. Jobs beyond
        the number of independent unit groups come back empty and their
        `map` is a no-op. A *stitched* segmentation additionally requires
        `prepare_jobs(n_jobs)` to have created the scratch root; each tile
        banks into its own array, so banking adds no conflicts and a
        stitched iterator splits exactly as a plain map would. Read-only
        iterators require
        `prepare_jobs` too, and distribute through their partial verbs —
        `measure_to_partial` / `detect_to_partial` per job, `merge_partials`
        as the gather — because their joins (coalesce, NMS) are global.
        `write_conflict_components` exposes the grouping this partition is
        built from.
        """
        if self._partition is not None:
            _, current_index, current_n = self._partition
            raise NgioValueError(
                f"This iterator is already restricted to partition "
                f"{current_index}/{current_n}; partitions do not nest. "
                "Apply `for_job` once, to the full iterator."
            )
        if n_jobs < 1:
            raise NgioValueError(f"n_jobs must be >= 1, got {n_jobs}.")
        if not 0 <= job_index < n_jobs:
            raise NgioValueError(
                f"job_index must be in [0, {n_jobs}), got {job_index}."
            )
        self._require_splittable()
        self._validate_job_context(n_jobs)
        units = list(self._numpy_units_generator())
        partition = partition_components(write_conflict_components(units), n_jobs)
        restricted = self._new_from_rois(self.rois)
        restricted._partition = (
            frozenset(partition[job_index]),
            job_index,
            n_jobs,
        )
        return restricted

    def prepare_jobs(self, n_jobs: int) -> list[JobArgs]:
        """Set up a distributed run and return its parallelization list.

        The init step of the three-phase distributed recipe (init → parallel
        jobs → consolidate, the shape schedulers like Fractal run natively):
        it computes the partition, performs whatever setup the iterator needs
        — always **clearing any stale scratch state from earlier runs first**,
        so a crashed or superseded run can never leak into this one — and
        returns one `JobArgs` per non-empty partition, ready to become the
        scheduler's parallelization list:

        ```python
        # init task
        args_list = iterator.prepare_jobs(n_jobs=4)
        # -> [{"job_index": 0, "n_jobs": 4}, ...]

        # parallel task, per entry
        iterator.for_job(**args).map(func)

        # consolidate task, after all jobs
        iterator.finalize()
        ```

        Empty partitions are dropped from the list — no task is submitted
        for a share with nothing to do. For plain writing iterators this
        call is optional (there is nothing to set up; `for_job` alone
        works); a *stitching* segmentation requires it, because the scratch
        band arrays must exist — race-free — before any job banks into them.
        """
        if self._partition is not None:
            raise NgioValueError(
                "prepare_jobs belongs to the unrestricted iterator; this one "
                "is already restricted to a partition."
            )
        if n_jobs < 1:
            raise NgioValueError(f"n_jobs must be >= 1, got {n_jobs}.")
        self._require_splittable()
        units = list(self._numpy_units_generator())
        partition = partition_components(write_conflict_components(units), n_jobs)
        self._prepare_distributed(n_jobs)
        return [
            JobArgs(job_index=job_index, n_jobs=n_jobs)
            for job_index, indices in enumerate(partition)
            if indices
        ]

    def _prepare_distributed(self, n_jobs: int) -> None:
        """Per-class setup for a distributed run; the base has none.

        Implementations must clear their scratch state before recreating it:
        `prepare_jobs` is the one moment a distributed run may destroy
        leftovers, and a clean slate here is what makes stale or corrupted
        state from a failed run unreachable.
        """

    def _validate_job_context(self, n_jobs: int) -> None:
        """Hook for `for_job`-time requirements; the base has none.

        Subclasses whose jobs depend on prepared shared state (the stitch
        scratch) verify it here, so a missing `prepare_jobs` fails at
        `for_job` with a clear message instead of mid-map.
        """

    def _plan_fingerprint(self, n_jobs: int) -> str:
        """Deterministic identity of a distributed plan.

        Built from geometry — the write regions, the output array's shape
        and chunks, the halo, `n_jobs` — never from ROI *names*, which are
        not stable across processes (duplicate names are de-duplicated with
        per-process suffixes). Every process that builds the same iterator
        computes the same fingerprint, so a job or a consolidate step can
        prove it is talking about the same plan the init step prepared.
        """
        payload: list[str] = [
            self.__class__.__name__,
            f"n_jobs={n_jobs}",
            f"halo={sorted(self._halo.items())}",
        ]
        output = self.output_image
        if output is not None:
            payload.append(f"shape={output.zarr_array.shape}")
            payload.append(f"chunks={output.zarr_array.chunks}")
        regions = self._touched_write_regions()
        if regions is None:
            # Read-only iterators: the read regions are the plan's geometry
            # (halo included, via the grown read ROI each getter covers).
            regions = [
                self.build_numpy_getter(roi).slicing_ops.normalized_slicing_tuple
                for roi in self.rois
            ]
        payload.append(f"regions={regions}")
        payload.extend(self._fingerprint_extras())
        return hashlib.sha256("|".join(payload).encode()).hexdigest()

    def _fingerprint_extras(self) -> tuple[str, ...]:
        """Extra class-specific terms for `_plan_fingerprint`; base: none."""
        return ()

    def _require_unrestricted_finalize(self) -> None:
        """Refuse `finalize` on a partition slice: the gather is global.

        Called by the writing iterators' `finalize` implementations. A slice
        holds one job's share; finalizing from it would resolve the pyramid
        against a partially-written level 0. Rebuild the iterator without
        `for_job` and finalize once, after every job completes.
        """
        if self._partition is not None:
            _, job_index, n_jobs = self._partition
            raise NgioValueError(
                f"finalize() on partition {job_index}/{n_jobs}: the gather "
                "step is global and belongs to the unrestricted iterator. "
                "Rebuild it without `for_job` and call `finalize()` "
                "once, after every job has completed."
            )

    def map(
        self,
        func: Callable[[NumpyPipeType], NumpyPipeType],
        *,
        mapper: MapperProtocol[NumpyPipeType, NumpyPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels and write it back.

        Args:
            func: The transformation. Under a parallel mapper it runs on
                worker threads (or processes) and must be safe there.
            mapper: How the units are scheduled. `None` (the default) is
                serial — parallel writes stay explicit opt-in. Pass
                `ThreadedMapper()` or `ProcessMapper()` to fan out; each
                sizes its own pool from its `max_workers` argument, and both
                schedule the units into conflict-free waves (`plan_waves`).
        """
        if mapper is None:
            mapper = BasicMapper[NumpyPipeType, NumpyPipeType]()
        units = list(self._numpy_units_generator())
        self._require_writable_units(units)
        mapper(func, units)
        # A partition slice runs only its share; the finalize (the one global
        # step) belongs to the unrestricted iterator, once every job is done.
        if self._partition is None:
            self.finalize()

    def map_as_numpy(
        self,
        func: Callable[[NumpyPipeType], NumpyPipeType],
        *,
        mapper: MapperProtocol[NumpyPipeType, NumpyPipeType] | None = None,
    ) -> None:
        """Alias for `map()`."""
        return self.map(func, mapper=mapper)

    def _require_serial_dask_mapper(
        self, mapper: "MapperProtocol[DaskPipeType, R] | None"
    ) -> "MapperProtocol[DaskPipeType, R]":
        """The dask verbs run serially; reject any parallel mapper.

        `store_dask` scopes a process-global dask config option, so
        concurrent dask writers are unsafe by construction.
        """
        if mapper is None:
            return BasicMapper[DaskPipeType, R]()
        if isinstance(mapper, BasicMapper):
            return mapper
        raise NgioValueError(
            "The dask verbs run serially: dask schedules its own workers, "
            f"and the write path is not safe under {type(mapper).__name__}. "
            "Pass no mapper here, or use the numpy verbs with a parallel "
            "mapper."
        )

    @deprecated(
        replacement="map() / map_as_numpy(mapper=...)",
        removed_in="1.2",
    )
    def map_as_dask(
        self,
        func: Callable[[DaskPipeType], DaskPipeType],
        *,
        mapper: MapperProtocol[DaskPipeType, DaskPipeType] | None = None,
    ) -> None:
        """Apply a transformation function to the ROI pixels and write it back.

        Deprecated: removed in ngio=1.2.

        Runs serially: the dask pipes are already executed by dask's own
        scheduler, and the dask write path (`store_dask`) scopes a global
        dask config option, so concurrent callers are unsafe. A parallel
        `mapper` raises.
        """
        _mapper = self._require_serial_dask_mapper(mapper)
        units = list(self._dask_units_generator())
        self._require_writable_units(units)
        _mapper(func, units)
        if self._partition is None:
            self.finalize()

    def reduce(
        self,
        func: Callable[[NumpyPipeType], R],
        *,
        mapper: MapperProtocol[NumpyPipeType, R] | None = None,
    ) -> list[R]:
        """Apply a function to every ROI and collect the results without writing.

        Units are built read-only even on writable iterators: nothing is
        written and `finalize` does not run.

        Args:
            func: The function to apply; under a parallel mapper it must be
                safe on worker threads (or processes).
            mapper: How the units are scheduled. `None` (the default) is
                serial; pass `ThreadedMapper()` or `ProcessMapper()` to fan
                out, sized by their own `max_workers` argument.

        Returns:
            One result per ROI, in ROI order: `results[i]` corresponds to
            `self.rois[i]`, regardless of the mapper's execution order.
        """
        self._require_no_halo_on_readonly_verbs()
        if mapper is None:
            mapper = BasicMapper[NumpyPipeType, R]()
        return mapper(func, list(self._numpy_units_generator(with_setters=False)))

    def reduce_as_numpy(
        self,
        func: Callable[[NumpyPipeType], R],
        *,
        mapper: MapperProtocol[NumpyPipeType, R] | None = None,
    ) -> list[R]:
        """Alias for `reduce()`."""
        return self.reduce(func, mapper=mapper)

    @deprecated(
        replacement="reduce() / reduce_as_numpy(mapper=...)",
        removed_in="1.2",
    )
    def reduce_as_dask(
        self,
        func: Callable[[DaskPipeType], R],
        *,
        mapper: MapperProtocol[DaskPipeType, R] | None = None,
    ) -> list[R]:
        """Apply a function to every ROI and collect the results without writing.

        Deprecated: removed in ngio=1.2.

        Units are built read-only even on writable iterators: nothing is
        written and `finalize` does not run. Runs serially — the units hand
        `func` *lazy* dask arrays, so the per-unit work is graph
        construction, not IO. A parallel `mapper` raises (see `map_as_dask`).

        Returns:
            One result per ROI, in ROI order: `results[i]` corresponds to
            `self.rois[i]`, regardless of the mapper's execution order.
        """
        self._require_no_halo_on_readonly_verbs()
        _mapper = self._require_serial_dask_mapper(mapper)
        return _mapper(func, list(self._dask_units_generator(with_setters=False)))

    def check_if_regions_overlap(self) -> bool:
        """Check if any of the ROIs overlap logically.

        If two ROIs cover the same pixel, they are considered to overlap.
        This does not consider chunking or other storage details. Measured
        on the *read* regions — with a halo, neighbouring reads overlap by
        construction. For write safety, `check_if_write_units_overlap` is
        the right question.

        Returns:
            `True` if any ROIs overlap.
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

    def check_if_write_units_overlap(self) -> bool:
        """Check if any two ROIs write into the same write unit of the output.

        Measured on the write target: slicing tuples come from the setters and
        the grid is the output array's write granularity — the shard shape when
        the output is sharded (writes are atomic per shard object), the chunk
        shape otherwise. Two ROIs sharing a write unit make concurrent writes
        unsafe: the read-modify-write of that unit can lose data. The parallel
        mappers schedule such ROIs into separate waves, so this check answers
        "will my map run as a single fully-parallel wave?". A read-only
        iterator has no write hazard and always returns `False`.

        This is O(n^2) in the number of ROIs; avoid calling it repeatedly in a
        loop.

        Returns:
            bool: True if any ROIs overlap in chunks, False otherwise.
        """
        if len(self.rois) < 2:
            # Less than 2 ROIs cannot overlap
            return False

        footprints = (
            compute_write_footprint(setter)
            for setter in self._numpy_setters_generator()
            if setter is not None
        )
        non_empty = (footprint for footprint in footprints if footprint is not None)
        return any(chunk_rects_intersect(fi, fj) for fi, fj in _pairs_stream(non_empty))

    def require_no_write_units_overlap(self) -> None:
        """Ensure that the ROIs do not share write units on the output.

        The strict opt-in gate: the parallel mappers no longer refuse shared
        write units on their own (they wave-schedule around them), so call
        this to insist on a single-wave tiling instead.
        """
        if self.check_if_write_units_overlap():
            raise NgioValueError("Some ROIs share write units on the output.")
