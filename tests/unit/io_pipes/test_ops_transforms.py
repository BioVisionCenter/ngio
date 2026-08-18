import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_empty_ome_zarr
from ngio.io_pipes import (
    AxesOps,
    DaskTransformProtocol,
    NumpyTransformProtocol,
    SlicingOps,
)
from ngio.io_pipes._zoom_transform import BaseZoomTransform
from ngio.utils import NgioValueError


class RecordingTransform:
    """Numpy-only transform that records the order it is applied in."""

    def __init__(self, name: str, calls: list[tuple[str, str]]):
        self.name = name
        self.calls = calls

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        self.calls.append(("get", self.name))
        return array

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        self.calls.append(("set", self.name))
        return array


class DaskOnlyTransform:
    """Dask-only transform, without the numpy half of the protocol."""

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        return array

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        return array


def _make_image(shape=(16, 16), pixelsize=1.0):
    return create_empty_ome_zarr(
        store=MemoryStore(),
        shape=shape,
        axes_names="yx",
        pixelsize=pixelsize,
        levels=1,
    ).get_image()


def test_protocol_membership():
    calls: list[tuple[str, str]] = []
    numpy_only = RecordingTransform("a", calls)
    dask_only = DaskOnlyTransform()
    zoom = BaseZoomTransform(
        input_dimensions=_make_image().dimensions,
        target_dimensions=_make_image().dimensions,
    )

    assert isinstance(numpy_only, NumpyTransformProtocol)
    assert not isinstance(numpy_only, DaskTransformProtocol)
    assert isinstance(dask_only, DaskTransformProtocol)
    assert not isinstance(dask_only, NumpyTransformProtocol)
    # The concrete ngio transform implements both halves
    assert isinstance(zoom, NumpyTransformProtocol)
    assert isinstance(zoom, DaskTransformProtocol)


def test_public_import_locations():
    from ngio.io_pipes import (  # noqa: F401
        AxesOps,
        DaskTransformProtocol,
        NumpyTransformProtocol,
        TransformProtocol,
    )
    from ngio.transforms import (  # noqa: F401
        AxesOps,
        DaskTransformProtocol,
        NumpyTransformProtocol,
        SlicingOps,
        TransformProtocol,
        ZoomTransform,
    )


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


def test_numpy_only_transform_roundtrip():
    calls: list[tuple[str, str]] = []
    transform = RecordingTransform("a", calls)
    image = _make_image()

    array = image.get_as_numpy(transforms=[transform])
    image.set_array(patch=array, transforms=[transform])
    assert calls == [("get", "a"), ("set", "a")]


def test_numpy_only_transform_through_segmentation_iterator():
    from ngio import SegmentationIterator

    calls: list[tuple[str, str]] = []
    transform = RecordingTransform("a", calls)

    omezarr = create_empty_ome_zarr(
        store=MemoryStore(),
        shape=(16, 16),
        axes_names="yx",
        pixelsize=1.0,
        levels=1,
    )
    label = omezarr.derive_label("mask")
    iterator = SegmentationIterator(
        input_image=omezarr.get_image(),
        output_label=label,
        input_transforms=[transform],
        output_transforms=[transform],
    )
    iterator.map_as_numpy(lambda x: np.ones_like(x, dtype=np.uint8))
    assert ("get", "a") in calls
    assert ("set", "a") in calls


def test_numpy_only_transform_rejected_by_dask_pipes():
    image = _make_image()
    calls: list[tuple[str, str]] = []
    numpy_only = RecordingTransform("a", calls)

    with pytest.raises(NgioValueError, match="RecordingTransform"):
        image.get_as_dask(transforms=[numpy_only])

    patch = da.zeros(image.shape, dtype=image.zarr_array.dtype)
    with pytest.raises(NgioValueError, match="dask transform protocol"):
        image.set_array(patch=patch, transforms=[numpy_only])
    assert calls == []


def test_dask_only_transform_rejected_by_numpy_pipes():
    image = _make_image()
    dask_only = DaskOnlyTransform()

    with pytest.raises(NgioValueError, match="DaskOnlyTransform"):
        image.get_as_numpy(transforms=[dask_only])

    patch = np.zeros(image.shape, dtype=image.zarr_array.dtype)
    with pytest.raises(NgioValueError, match="numpy transform protocol"):
        image.set_array(patch=patch, transforms=[dask_only])


def test_dask_only_transform_accepted_by_dask_pipes():
    image = _make_image()
    dask_only = DaskOnlyTransform()

    array = image.get_as_dask(transforms=[dask_only])
    image.set_array(patch=array, transforms=[dask_only])
