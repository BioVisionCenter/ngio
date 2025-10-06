import pytest

from ngio.common import Dimensions
from ngio.ome_zarr_meta import AxesHandler, Dataset
from ngio.ome_zarr_meta.ngio_specs import Axis
from ngio.utils import NgioValueError


@pytest.mark.parametrize(
    "axes_names",
    [
        ["x", "y", "z", "c"],
        ["x", "y", "c"],
        ["z", "c", "x", "y"],
        ["t", "z", "c", "x", "y"],
        ["x", "y", "z", "t"],
    ],
)
def test_dimensions(axes_names):
    axes = [Axis(name=name) for name in axes_names]
    canonic_dim_dict = dict(zip("tczyx", (2, 3, 4, 5, 6), strict=True))
    dim_dict = {ax: canonic_dim_dict.get(ax, 1) for ax in axes_names}

    shape = tuple(dim_dict.get(ax) for ax in axes_names)
    shape = tuple(s for s in shape if s is not None)

    ax_mapper = AxesHandler(axes=axes)

    ds = Dataset(
        path="0",
        axes_handler=ax_mapper,
        scale=[1.0] * len(axes_names),
        translation=[0.0] * len(axes_names),
    )

    dims = Dimensions(shape=shape, chunks=shape, dataset=ds)

    assert isinstance(dims.__repr__(), str)

    for ax, s in dim_dict.items():
        assert dims.get(ax) == s

    assert dims.axes_handler.get_index("x") == axes_names.index("x")
    assert dims.axes_handler.get_index("-") is None

    if dim_dict.get("z", 1) > 1:
        assert dims.is_3d

    if dim_dict.get("c", 1) > 1:
        assert dims.is_multi_channels

    if dim_dict.get("t", 1) > 1:
        assert dims.is_time_series

    if dim_dict.get("z", 1) > 1 and dim_dict.get("t", 1) > 1:
        assert dims.is_3d_time_series

    if dim_dict.get("z", 1) == 1 and dim_dict.get("t", 1) == 1:
        assert dims.is_2d

    if dim_dict.get("z", 1) == 1 and dim_dict.get("t", 1) > 1:
        assert dims.is_2d_time_series

    assert dims.shape == shape


def test_dimensions_error():
    axes = [Axis(name="x"), Axis(name="y")]
    shape = (1, 2, 3)
    ax_handler = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_handler,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )

    with pytest.raises(NgioValueError):
        Dimensions(shape=shape, chunks=shape, dataset=ds)

    shape = (3, 4)
    dims = Dimensions(shape=shape, chunks=shape, dataset=ds)

    assert dims.get("c", default=None) is None
    assert not dims.is_3d
    assert not dims.is_multi_channels
    assert not dims.is_time_series
    assert not dims.is_3d_time_series
    assert not dims.is_2d_time_series


def test_dimensions_checks():
    axes = [Axis(name="c"), Axis(name="z"), Axis(name="x"), Axis(name="y")]
    shape = (3, 4, 8, 8)
    ax_handler = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_handler,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )
    dims_ref = Dimensions(shape=shape, chunks=shape, dataset=ds)

    # Test with rescalable dimensions
    # Axes matches, dimensions do not match, but are rescalable
    axes = [Axis(name="c"), Axis(name="z"), Axis(name="x"), Axis(name="y")]
    shape = (3, 4, 16, 16)
    ax_handler = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_handler,
        scale=[1.0, 1.0, 0.5, 0.5],
        translation=[0.0] * len(axes),
    )
    dims_other = Dimensions(shape=shape, chunks=shape, dataset=ds)

    dims_ref.require_axes_match(dims_other)
    assert dims_ref.check_if_axes_match(dims_other)

    with pytest.raises(NgioValueError):
        dims_ref.require_dimensions_match(dims_other, allow_singleton=False)
    assert not dims_ref.check_if_dimensions_match(dims_other, allow_singleton=False)

    dims_ref.require_rescalable(dims_other)
    assert dims_ref.check_if_rescalable(dims_other)

    # Test with non-matching axes
    # Axes do not match

    axes = [Axis(name="x"), Axis(name="y")]
    shape = (8, 8)
    ax_handler = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_handler,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )
    dims_other = Dimensions(shape=shape, chunks=shape, dataset=ds)

    with pytest.raises(NgioValueError):
        dims_ref.require_axes_match(dims_other)
    assert not dims_ref.check_if_axes_match(dims_other)

    # Axes match, dimensions do (with singleton allowed)
    axes = [Axis(name="z"), Axis(name="x"), Axis(name="y")]
    shape = (1, 8, 8)
    ax_handler = AxesHandler(axes=axes)
    ds = Dataset(
        path="0",
        axes_handler=ax_handler,
        scale=[1.0] * len(axes),
        translation=[0.0] * len(axes),
    )
    dims_other = Dimensions(shape=shape, chunks=shape, dataset=ds)
    dims_ref.require_axes_match(dims_other)
    assert dims_ref.check_if_axes_match(dims_other)

    assert not dims_ref.check_if_dimensions_match(dims_other, allow_singleton=False)
    dims_ref.require_dimensions_match(dims_other, allow_singleton=True)
    assert dims_ref.check_if_dimensions_match(dims_other, allow_singleton=True)
