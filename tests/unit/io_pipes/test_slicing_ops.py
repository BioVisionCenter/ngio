import pytest
import zarr

from ngio.common import Dimensions
from ngio.io_pipes._ops_slices import (
    build_slicing_ops,
    get_slice_as_dask,
    get_slice_as_numpy,
    set_slice_as_dask,
    set_slice_as_numpy,
)
from ngio.io_pipes._ops_slices_utils import (
    check_elem_intersection,
    check_if_regions_overlap,
)
from ngio.ome_zarr_meta import AxesHandler, Dataset
from ngio.ome_zarr_meta.ngio_specs import Axis


@pytest.mark.parametrize(
    "input_axes,output_axes,input_shape,output_shape,slicing_dict",
    [
        ("tczyx", "zyx", (1, 1, 10, 11, 12), (10, 11, 12), {"t": 0, "c": 0}),
        ("czyx", "czyx", (3, 10, 11, 12), (2, 10, 11, 12), {"c": (0, 2)}),
        ("czyx", "czy", (3, 10, 11, 12), (2, 10, 11), {"c": (0, 2), "x": 4}),
        ("zyx", "zyx", (10, 11, 12), (10, 11, 12), {"t": slice(None), "c": 0}),
    ],
)
def test_slicing_ops_base(
    input_axes: str,
    output_axes: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    slicing_dict: dict[str, int | slice | tuple[int, ...]],
):
    axes = [Axis(name=name) for name in input_axes]

    ax_mapper = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_mapper,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )
    dims = Dimensions(shape=input_shape, chunks=input_shape, dataset=ds)

    slicing_ops = build_slicing_ops(
        dimensions=dims,
        slicing_dict=slicing_dict,  # type: ignore[arg-type]
        remove_channel_selection=False,
    )
    assert slicing_ops.slice_axes == tuple(output_axes)
    assert slicing_ops.slice_chunks() == {(0,) * len(slicing_ops.on_disk_axes)}

    # Numpy
    zarr_data = zarr.zeros(shape=input_shape, chunks=input_shape, dtype="uint8")
    data = get_slice_as_numpy(zarr_data, slicing_ops=slicing_ops)
    assert data.shape == output_shape
    set_slice_as_numpy(zarr_array=zarr_data, patch=data, slicing_ops=slicing_ops)

    # Dask
    zarr_data = zarr.zeros(shape=input_shape, chunks=input_shape, dtype="uint8")
    data = get_slice_as_dask(zarr_data, slicing_ops=slicing_ops)
    assert data.shape == output_shape
    set_slice_as_dask(zarr_array=zarr_data, patch=data, slicing_ops=slicing_ops)


def test_chunk_slice():
    input_axes = "tczyx"
    input_shape = (2, 3, 10, 11, 12)
    chunk_shape = (1, 1, 10, 11, 12)
    axes = [Axis(name=name) for name in input_axes]

    ax_mapper = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_mapper,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )
    dims = Dimensions(shape=input_shape, chunks=chunk_shape, dataset=ds)

    slicing_dict = {"t": 0, "c": (0, 2)}
    slicing_ops1 = build_slicing_ops(
        dimensions=dims,
        slicing_dict=slicing_dict,  # type: ignore[arg-type]
        remove_channel_selection=False,
    )
    assert slicing_ops1.slice_chunks() == {(0, 0, 0, 0, 0), (0, 2, 0, 0, 0)}

    slicing_dict = {"t": 1, "c": (0, 2)}
    slicing_ops2 = build_slicing_ops(
        dimensions=dims,
        slicing_dict=slicing_dict,  # type: ignore[arg-type]
        remove_channel_selection=False,
    )
    assert slicing_ops2.slice_chunks() == {(1, 0, 0, 0, 0), (1, 2, 0, 0, 0)}

    # Check for overlapping regions
    assert not check_if_regions_overlap(
        [slicing_ops1.normalized_slicing_tuple, slicing_ops2.normalized_slicing_tuple]
    )

    # Repeat with bigger tuples as different element
    slicing_dict = {"t": 0, "c": (0, 2)}
    slicing_ops2 = build_slicing_ops(
        dimensions=dims,
        slicing_dict=slicing_dict,  # type: ignore[arg-type]
        remove_channel_selection=False,
    )

    # Check for overlapping regions
    assert check_if_regions_overlap(
        [slicing_ops1.normalized_slicing_tuple, slicing_ops2.normalized_slicing_tuple]
    )


def test_regions_overlap_sweep_agrees_with_all_pairs():
    """The sort-and-sweep must answer exactly like the pairwise scan."""
    import random

    from ngio.io_pipes._ops_slices_utils import (
        _pairs_stream,
        check_slicing_tuple_intersection,
    )

    rng = random.Random(0)

    def brute(regions):
        return any(
            check_slicing_tuple_intersection(a, b) for a, b in _pairs_stream(regions)
        )

    for _ in range(50):
        regions = []
        for _ in range(rng.randint(2, 12)):
            start_y, start_x = rng.randrange(0, 40), rng.randrange(0, 40)
            regions.append(
                (
                    rng.randrange(0, 2),
                    slice(start_y, start_y + rng.randrange(1, 12)),
                    slice(start_x, start_x + rng.randrange(1, 12)),
                )
            )
        assert check_if_regions_overlap(regions) == brute(regions), regions


def test_regions_overlap_handles_list_only_axes():
    """List selections are not intervals; the fallback must still answer."""
    assert check_if_regions_overlap([([0, 1],), ([2, 3],)]) is False
    assert check_if_regions_overlap([([0, 1],), ([1, 2],)]) is True


def test_check_elem_intersection_step_not_implemented():
    with pytest.raises(NotImplementedError):
        check_elem_intersection(slice(0, 5, 2), slice(1, 3))
    with pytest.raises(NotImplementedError):
        check_elem_intersection(slice(1, 3), slice(0, 5, 2))
