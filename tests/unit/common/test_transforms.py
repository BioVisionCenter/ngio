import dask.array as da
import numpy as np

from ngio.common.transforms import ZoomTransform
from ngio.ome_zarr_meta.ngio_specs import SlicingOps


def test_zoom_transform():
    zoom = ZoomTransform(scale=(1, 2, 2), order=0)
    assert zoom.scale == (1, 2, 2)
    assert zoom.inv_scale == (1.0, 0.5, 0.5)

    x = np.ones((1, 10, 10))
    x_zoomed = zoom.apply_numpy_transform(
        array=x,
        slicing_ops=SlicingOps(),
    )

    assert x_zoomed.shape == (1, 20, 20)

    x_inverse = zoom.apply_inverse_numpy_transform(
        array=x_zoomed,
        slicing_ops=SlicingOps(),
    )
    assert x_inverse.shape == (1, 10, 10)

    x_dask = da.from_array(x)
    x_zoomed_dask = zoom.apply_dask_transform(
        array=x_dask,
        slicing_ops=SlicingOps(),
    )
    assert x_zoomed_dask.shape == (1, 20, 20)
    x_inverse_dask = zoom.apply_inverse_dask_transform(
        array=x_zoomed_dask,
        slicing_ops=SlicingOps(),
    )
    assert x_inverse_dask.shape == (1, 10, 10)


def test_zoom_from_dimensions():
    from ngio.common import Dimensions
    from ngio.ome_zarr_meta import AxesMapper
    from ngio.ome_zarr_meta.ngio_specs import Axis

    original_axes = [Axis(on_disk_name=ax) for ax in ("t", "z", "y", "x")]
    target_axes = [Axis(on_disk_name=ax) for ax in ("z", "y", "x")]

    original_dim = Dimensions(
        shape=(10, 3, 10, 10), axes_mapper=AxesMapper(original_axes)
    )
    target_dim = Dimensions(shape=(3, 5, 5), axes_mapper=AxesMapper(target_axes))

    zoom = ZoomTransform.from_dimensions(
        original_dimension=original_dim,
        target_dimension=target_dim,
        order=1,
    )

    assert np.allclose(zoom.scale, (1, 1, 0.5, 0.5))
