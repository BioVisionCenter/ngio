import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import Roi, create_empty_ome_zarr
from ngio.common import Dimensions
from ngio.io_pipes._match_shape import numpy_match_shape
from ngio.io_pipes._ops_slices import build_slicing_ops
from ngio.io_pipes._zoom_transform import BaseZoomTransform
from ngio.ome_zarr_meta import AxesHandler, Dataset
from ngio.ome_zarr_meta.ngio_specs import AxesSetup, Axis
from ngio.transforms import ZoomTransform
from ngio.utils import NgioValueError


def _make_dims(
    axes: list[str],
    shape: tuple[int, ...],
    axes_setup: AxesSetup | None = None,
) -> Dimensions:
    axes_list = [Axis(name=name) for name in axes]
    handler = AxesHandler(axes=axes_list, axes_setup=axes_setup)
    dataset = Dataset(
        path="0",
        axes_handler=handler,
        scale=[1.0] * len(axes_list),
        translation=[0.0] * len(axes_list),
    )
    return Dimensions(shape=shape, chunks=shape, dataset=dataset)


##############################################################
# _match_shape.py: _check_axes error branches
##############################################################


def test_match_shape_array_axes_length_mismatch():
    with pytest.raises(NgioValueError, match="same number of dimensions"):
        numpy_match_shape(
            array=np.zeros((2, 2)),
            reference_shape=(2, 2),
            array_axes=["z", "y", "x"],
            reference_axes=["y", "x"],
        )


def test_match_shape_reference_axes_length_mismatch():
    with pytest.raises(NgioValueError, match="same number of dimensions"):
        numpy_match_shape(
            array=np.zeros((2, 2)),
            reference_shape=(2, 2, 2),
            array_axes=["y", "x"],
            reference_axes=["y", "x"],
        )


def test_match_shape_axes_not_subset():
    with pytest.raises(NgioValueError, match="not a subset"):
        numpy_match_shape(
            array=np.zeros((2, 2)),
            reference_shape=(2, 2),
            array_axes=["w", "x"],
            reference_axes=["y", "x"],
        )


def test_match_shape_more_dims_than_reference():
    # Duplicated axis names make the subset check pass while the array
    # still has more dimensions than the reference
    with pytest.raises(NgioValueError, match="more dimensions"):
        numpy_match_shape(
            array=np.zeros((2, 2)),
            reference_shape=(2,),
            array_axes=["x", "x"],
            reference_axes=["x"],
        )


##############################################################
# _zoom_transform.py: _normalize_shape and dask set transform
##############################################################


def test_zoom_normalize_shape_branches():
    dims = _make_dims(["y", "x"], (8, 8))
    zoom = BaseZoomTransform(input_dimensions=dims, target_dimensions=dims)

    # Integer slicing selects a single element regardless of the scale
    assert zoom._normalize_shape(slice_=3, scale=2.0, max_dim=10) == 1
    # Open-ended slice: the stop defaults to the axis size
    assert zoom._normalize_shape(slice_=slice(2, None), scale=1.0, max_dim=10) == 8
    # List slicing scales with the number of selected elements
    assert zoom._normalize_shape(slice_=[0, 1, 2], scale=2.0, max_dim=10) == 6
    with pytest.raises(ValueError, match="Unsupported slice type"):
        zoom._normalize_shape(slice_="bad", scale=1.0, max_dim=10)  # ty: ignore[invalid-argument-type]


def test_zoom_set_as_dask_transform():
    target_img = create_empty_ome_zarr(
        store=MemoryStore(),
        shape=(100, 100),
        axes_names="yx",
        pixelsize=1.0,
        levels=1,
    ).get_image()
    input_img = create_empty_ome_zarr(
        store=MemoryStore(),
        shape=(50, 50),
        axes_names="yx",
        pixelsize=2.0,
        levels=1,
    ).get_image()

    zoom = ZoomTransform(input_image=input_img, target_image=target_img)
    roi = Roi.from_values(name=None, slices={"x": (10, 30), "y": (10, 30)})

    target_data = target_img.get_roi_as_dask(roi=roi)
    rescaled = input_img.get_roi_as_dask(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled.shape
    # Setting a dask patch triggers the inverse (set) dask zoom transform
    input_img.set_roi(roi=roi, patch=rescaled, transforms=[zoom])

    # Same round trip with a numpy patch for the numpy set transform
    rescaled_np = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_np.shape
    input_img.set_roi(roi=roi, patch=rescaled_np, transforms=[zoom])


##############################################################
# _ops_slices.py: error branches and slicing normalization
##############################################################


def test_slicing_int_out_of_bounds():
    dims = _make_dims(["c", "y", "x"], (3, 8, 8))

    ops = build_slicing_ops(dimensions=dims, slicing_dict={"y": 100})
    with pytest.raises(NgioValueError, match="out of bounds"):
        _ = ops.normalized_slicing_tuple

    ops = build_slicing_ops(dimensions=dims, slicing_dict={"y": -1})
    with pytest.raises(NgioValueError, match="out of bounds"):
        _ = ops.normalized_slicing_tuple

    # Out-of-bounds index inside a non-contiguous list selection
    ops = build_slicing_ops(dimensions=dims, slicing_dict={"y": (0, 100)})
    with pytest.raises(NgioValueError, match="out of bounds"):
        _ = ops.normalized_slicing_tuple


def test_slicing_empty_sequence():
    dims = _make_dims(["y", "x"], (8, 8))
    with pytest.raises(NgioValueError, match="empty sequences"):
        build_slicing_ops(dimensions=dims, slicing_dict={"y": ()})


def test_slicing_sequence_of_numpy_ints():
    dims = _make_dims(["c", "y", "x"], (3, 8, 8))
    # numpy integers are not `int` instances and must be coerced
    ops = build_slicing_ops(
        dimensions=dims,
        slicing_dict={"c": (np.uint8(0), np.uint8(2))},  # ty: ignore[invalid-argument-type]
    )
    assert ops.slicing_tuple[0] == [0, 2]

    # Contiguous numpy integers are converted to a slice
    ops = build_slicing_ops(
        dimensions=dims,
        slicing_dict={"c": (np.uint8(0), np.uint8(1))},  # ty: ignore[invalid-argument-type]
    )
    assert ops.slicing_tuple[0] == slice(0, 2)


def test_slicing_sequence_of_invalid_values():
    dims = _make_dims(["y", "x"], (8, 8))
    with pytest.raises(NgioValueError, match="Invalid value"):
        build_slicing_ops(dimensions=dims, slicing_dict={"y": ("a", "b")})  # ty: ignore[invalid-argument-type]


def test_slicing_channel_selection_kept_on_multichannel():
    dims = _make_dims(["c", "y", "x"], (3, 8, 8))
    ops = build_slicing_ops(
        dimensions=dims,
        slicing_dict={"c": 0},
        remove_channel_selection=True,
    )
    # Multi-channel images keep the channel selection
    assert ops.slicing_tuple[0] == 0


def test_slicing_channel_selection_removed_on_singleton():
    dims = _make_dims(["c", "y", "x"], (1, 8, 8))
    ops = build_slicing_ops(
        dimensions=dims,
        slicing_dict={"c": 0},
        remove_channel_selection=True,
    )
    # Singleton channel axis: the channel selection is dropped
    assert ops.slicing_tuple[0] == slice(None)


@pytest.mark.parametrize(
    "slice_",
    [
        slice(0, None),
        slice(None, 0),
        [0],
        None,
        0,
        slice(None),
        slice(0, 1),
    ],
)
def test_slicing_valid_virtual_axis(slice_):
    dims = _make_dims(["y", "x"], (8, 8))
    # "c" is not an on-disk axis, but trivial selections are allowed
    ops = build_slicing_ops(dimensions=dims, slicing_dict={"c": slice_})
    assert ops.slicing_tuple == (slice(None), slice(None))


@pytest.mark.parametrize("slice_", [5, slice(1, 3), [1], [0, 1]])
def test_slicing_invalid_virtual_axis(slice_):
    dims = _make_dims(["y", "x"], (8, 8))
    with pytest.raises(NgioValueError, match="Invalid axis selection"):
        build_slicing_ops(dimensions=dims, slicing_dict={"c": slice_})


def test_slicing_duplicate_axis():
    # "channel" is the on-disk name for the canonical "c" axis, so both
    # names resolve to the same axis
    dims = _make_dims(
        ["channel", "y", "x"],
        (3, 8, 8),
        axes_setup=AxesSetup(c="channel"),
    )
    with pytest.raises(NgioValueError, match="Duplicate axis"):
        build_slicing_ops(
            dimensions=dims,
            slicing_dict={"channel": 0, "c": 0},
        )


def test_slicing_explicit_none_value():
    dims = _make_dims(["y", "x"], (8, 8))
    ops = build_slicing_ops(dimensions=dims, slicing_dict={"y": None})
    assert ops.slicing_tuple == (slice(None), slice(None))
    # Querying an axis that is not on disk returns a full slice
    assert ops.get("t") == slice(None)


def test_slicing_invalid_value_type():
    dims = _make_dims(["y", "x"], (8, 8))
    with pytest.raises(NgioValueError, match="Invalid slice definition"):
        build_slicing_ops(dimensions=dims, slicing_dict={"y": 1.5})  # ty: ignore[invalid-argument-type]


def test_negative_value_in_slicing_sequence_raises_ngio_error():
    dims = _make_dims(["y", "x"], (10, 10))
    with pytest.raises(NgioValueError):
        build_slicing_ops(
            dimensions=dims,
            slicing_dict={"y": [-1, 0]},
        )
