import pytest

from ngio.common import Dimensions
from ngio.ome_zarr_meta import AxesMapper
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
    axes = [Axis(on_disk_name=name) for name in axes_names]
    canonic_dim_dict = dict(zip("tczyx", (2, 3, 4, 5, 6), strict=True))
    dim_dict = {ax: canonic_dim_dict.get(ax, 1) for ax in axes_names}

    shape = tuple(dim_dict.get(ax) for ax in axes_names)
    shape = tuple(s for s in shape if s is not None)

    ax_mapper = AxesMapper(on_disk_axes=axes)
    dims = Dimensions(shape=shape, axes_mapper=ax_mapper)

    assert isinstance(dims.__repr__(), str)

    for ax, s in dim_dict.items():
        assert dims.get(ax) == s

    assert dims.get_index("x") == axes_names.index("x")
    assert dims.get_index("-") is None

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
    axes = [Axis(on_disk_name="x"), Axis(on_disk_name="y")]
    shape = (1, 2, 3)

    with pytest.raises(NgioValueError):
        Dimensions(shape=shape, axes_mapper=AxesMapper(on_disk_axes=axes))

    shape = (3, 4)
    dims = Dimensions(shape=shape, axes_mapper=AxesMapper(on_disk_axes=axes))

    assert dims.get("c", default=None) is None
    assert not dims.is_3d
    assert not dims.is_multi_channels
    assert not dims.is_time_series
    assert not dims.is_3d_time_series
    assert not dims.is_2d_time_series
