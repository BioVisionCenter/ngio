import dask.array as da
import numpy as np
import pytest

from ngio.io_pipes._ops_axes import (
    _build_axes_ops,
    get_as_dask_axes_ops,
    get_as_numpy_axes_ops,
    get_as_sequence_axes_ops,
    set_as_dask_axes_ops,
    set_as_numpy_axes_ops,
    set_as_sequence_axes_ops,
)


@pytest.mark.parametrize(
    "input_axes,output_axes,input_shape,output_shape",
    [
        ("tczyx", "zyx", (1, 1, 10, 11, 12), (10, 11, 12)),
        ("czyx", "zcyx", (1, 10, 11, 12), (10, 1, 11, 12)),
        ("zyx", "tczyx", (10, 11, 12), (1, 1, 10, 11, 12)),
        ("czyx", "yxtc", (3, 1, 11, 12), (11, 12, 1, 3)),
    ],
)
def test_axes_base(input_axes, output_axes, input_shape, output_shape):
    input_axes = tuple(input_axes)
    output_axes = tuple(output_axes)
    ops = _build_axes_ops(input_axes, output_axes)

    # Round for tuples
    get_shape = get_as_sequence_axes_ops(input_shape, axes_ops=ops, default=1)
    assert tuple(get_shape) == output_shape
    set_shape = set_as_sequence_axes_ops(get_shape, axes_ops=ops, default=1)
    assert tuple(set_shape) == input_shape

    # Numpy
    input_data = np.zeros(input_shape)
    get_data = get_as_numpy_axes_ops(input_data, axes_ops=ops)
    assert get_data.shape == output_shape
    set_data = set_as_numpy_axes_ops(get_data, axes_ops=ops)
    assert set_data.shape == input_shape

    # Dask
    input_data = da.zeros(input_shape)
    get_data = get_as_dask_axes_ops(input_data, axes_ops=ops)
    assert get_data.shape == output_shape
    set_data = set_as_dask_axes_ops(get_data, axes_ops=ops)
    assert set_data.shape == input_shape
