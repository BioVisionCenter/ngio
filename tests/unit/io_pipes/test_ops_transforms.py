import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import Roi, create_empty_ome_zarr
from ngio.io_pipes import TransformContext
from ngio.io_pipes._zoom_transform import BaseZoomTransform
from ngio.utils import NgioDeprecationWarning, NgioValueError


class RecordingTransform:
    """Records the order it is applied in, and the ctx it received."""

    def __init__(self, name: str, calls: list[tuple[str, str]]):
        self.name = name
        self.calls = calls
        self.contexts: list[TransformContext] = []

    def on_get(self, array: np.ndarray, ctx: TransformContext) -> np.ndarray:
        self.calls.append(("get", self.name))
        self.contexts.append(ctx)
        return array

    def on_set(self, array: np.ndarray, ctx: TransformContext) -> np.ndarray:
        self.calls.append(("set", self.name))
        self.contexts.append(ctx)
        return array


class LegacyTransform:
    """The pre-1.1 contract, as implemented by fractal-tasks-utils."""

    def __init__(self, calls: list[str] | None = None):
        self.calls = calls if calls is not None else []

    def get_as_numpy_transform(self, array, slicing_ops, axes_ops):
        self.calls.append("get_numpy")
        return array

    def get_as_dask_transform(self, array, slicing_ops, axes_ops):
        raise NotImplementedError("no dask support")

    def set_as_numpy_transform(self, array, slicing_ops, axes_ops):
        self.calls.append("set_numpy")
        return array

    def set_as_dask_transform(self, array, slicing_ops, axes_ops):
        return array


class MaterializingTransform:
    """A buggy transform that silently computes a dask array."""

    def on_get(self, array, ctx: TransformContext):
        return np.asarray(array)

    def on_set(self, array, ctx: TransformContext):
        return np.asarray(array)


def _make_image(shape=(16, 16), pixelsize=1.0):
    return create_empty_ome_zarr(
        store=MemoryStore(),
        shape=shape,
        axes_names="yx",
        pixelsize=pixelsize,
        levels=1,
    ).get_image()


def test_protocol_membership_and_imports():
    from ngio.transforms import (  # noqa: F401
        AxesOps,
        SlicingOps,
        TransformContext,
        TransformProtocol,
        ZoomTransform,
    )

    calls: list[tuple[str, str]] = []
    assert isinstance(RecordingTransform("a", calls), TransformProtocol)
    assert not isinstance(LegacyTransform(), TransformProtocol)
    zoom = BaseZoomTransform(
        input_dimensions=_make_image().dimensions,
        target_dimensions=_make_image().dimensions,
    )
    assert isinstance(zoom, TransformProtocol)


def test_set_transforms_applied_in_reverse_order():
    calls: list[tuple[str, str]] = []
    chain = [RecordingTransform("a", calls), RecordingTransform("b", calls)]
    image = _make_image()

    image.get_as_numpy(transforms=chain)
    assert calls == [("get", "a"), ("get", "b")]

    calls.clear()
    patch = np.zeros(image.shape, dtype=image.zarr_array.dtype)
    # Reading applies a then b, so writing must invert the chain: b then a
    image.set_array(patch=patch, transforms=chain)
    assert calls == [("set", "b"), ("set", "a")]


def test_transform_context_contents():
    calls: list[tuple[str, str]] = []
    transform = RecordingTransform("a", calls)
    image = _make_image()

    image.get_as_numpy(transforms=[transform], axes_order=["x", "y"])
    ctx = transform.contexts[-1]
    assert ctx.axes == ("x", "y")
    assert ctx.roi is None

    # from_values slices are (start, length): x covers [2, 12)
    roi = Roi.from_values(name="r", slices={"x": (2, 10), "y": (4, 12)})
    image.get_roi_as_numpy(roi=roi, transforms=[transform])
    ctx = transform.contexts[-1]
    assert ctx.roi is not None and ctx.roi.name == "r"
    assert ctx.slicing.get("x") == slice(2, 12)


def test_io_pipe_context_is_transform_context():
    from ngio.io_pipes import IoPipeContext
    from ngio.transforms import TransformContext as TransformContextAlias

    assert TransformContext is IoPipeContext
    assert TransformContextAlias is IoPipeContext


def test_io_pipe_context_plumbing_fields():
    calls: list[tuple[str, str]] = []
    transform = RecordingTransform("a", calls)
    image = _make_image()

    image.get_as_numpy(transforms=[transform])
    ctx = transform.contexts[-1]
    assert ctx.zarr_array is image.zarr_array
    assert ctx.axes == tuple(ctx.axes_ops.output_axes)


def test_dual_backend_transform_roundtrip():
    calls: list[tuple[str, str]] = []
    transform = RecordingTransform("a", calls)
    image = _make_image()

    array = image.get_as_numpy(transforms=[transform])
    image.set_array(patch=array, transforms=[transform])
    assert calls == [("get", "a"), ("set", "a")]


def test_numpy_only_transform_on_dask_path_raises():
    image = _make_image()
    with pytest.raises(NgioValueError, match="MaterializingTransform"):
        # on_get returns a computed numpy array: the result-type check fires
        image.get_as_dask(transforms=[MaterializingTransform()]).compute()


def test_non_transform_rejected_at_construction():
    image = _make_image()
    with pytest.raises(NgioValueError, match="object"):
        image.get_as_numpy(transforms=[object()])


def test_legacy_transform_adapted_with_warning():
    image = _make_image()
    legacy = LegacyTransform()

    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        array = image.get_as_numpy(transforms=[legacy])
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        image.set_array(patch=array, transforms=[legacy])
    assert legacy.calls == ["get_numpy", "set_numpy"]


def test_legacy_transform_dask_still_raises_not_implemented():
    # fractal-style legacy transforms raise NotImplementedError on the dask
    # path; the adapter must preserve that behavior, not mask it. Transforms
    # run at graph-construction time, so the error fires on get_as_dask itself.
    image = _make_image()
    with (
        pytest.warns(NgioDeprecationWarning),
        pytest.raises(NotImplementedError, match="no dask support"),
    ):
        image.get_as_dask(transforms=[LegacyTransform()])


def test_legacy_zoom_equivalence():
    # A legacy identity chain must produce the same pixels as the new contract
    rng = np.random.default_rng(0)
    image = _make_image()
    image.set_array(patch=rng.integers(0, 255, size=(16, 16)).astype(np.uint16))

    calls: list[tuple[str, str]] = []
    via_new = image.get_as_numpy(transforms=[RecordingTransform("a", calls)])
    with pytest.warns(NgioDeprecationWarning):
        via_legacy = image.get_as_numpy(transforms=[LegacyTransform()])
    np.testing.assert_array_equal(via_new, via_legacy)


def test_zoom_transform_both_backends():
    target = create_empty_ome_zarr(
        store=MemoryStore(), shape=(32, 32), axes_names="yx", pixelsize=1.0, levels=1
    ).get_image()
    source = create_empty_ome_zarr(
        store=MemoryStore(), shape=(16, 16), axes_names="yx", pixelsize=2.0, levels=1
    ).get_image()
    zoom = BaseZoomTransform(
        input_dimensions=source.dimensions, target_dimensions=target.dimensions
    )

    up_np = source.get_as_numpy(transforms=[zoom])
    assert up_np.shape == (32, 32)
    up_da = source.get_as_dask(transforms=[zoom])
    assert isinstance(up_da, da.Array)
    assert up_da.shape == (32, 32)

    source.set_array(patch=up_np, transforms=[zoom])
    source.set_array(patch=up_da, transforms=[zoom])
