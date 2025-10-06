from math import ceil

import numpy as np


def _center_crop(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    # Center-crop the array `arr` along dimension `axis` to size `target`.
    # This assumes target < arr.shape[axis].
    n = arr.shape[axis]
    start = (n - target) // 2
    end = start + target
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(start, end)
    return arr[tuple(slc)]


def _tile_to(
    arr: np.ndarray, target: int, axis: int, label_mode: bool = False
) -> np.ndarray:
    # Tile the array `arr` along dimension `axis` to size `target`.
    # This assumes target > arr.shape[axis].
    n = arr.shape[axis]
    reps = ceil(target / n)

    tiles = []
    flip = False
    max_label = 0
    for _ in range(reps):
        if flip:
            t_arr = np.flip(arr, axis=axis)
        else:
            t_arr = 1 * arr
        if label_mode:
            # Remove duplicate labels
            t_arr = np.where(t_arr > 0, t_arr + max_label, 0)
            max_label = t_arr.max()
        tiles.append(t_arr)
        flip = not flip

    tiled = np.concatenate(tiles, axis=axis)

    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(0, target)
    return tiled[tuple(slc)]


def _fit_to_shape_2d(
    src: np.ndarray, out_shape: tuple[int, int], label_mode: bool = False
) -> np.ndarray:
    """Fit a 2D array to a target shape by center-cropping or tiling as necessary."""
    out_r, out_c = out_shape
    arr = src
    if out_r < arr.shape[0]:
        arr = _center_crop(arr, out_r, axis=0)
    else:
        arr = _tile_to(arr, out_r, axis=0, label_mode=label_mode)

    if out_c < arr.shape[1]:
        arr = _center_crop(arr, out_c, axis=1)
    else:
        arr = _tile_to(arr, out_c, axis=1, label_mode=label_mode)
    return arr


def fit_to_shape(
    arr: np.ndarray, out_shape: tuple[int, ...], ensure_unique_info: bool = False
) -> np.ndarray:
    """Fit a 2D array to a target shape.

    The x,y dimensions of `arr` are fitted to the last two dimensions of
    `out_shape` by center-cropping or tiling as necessary.
    The other dimensions are broadcasted as necessary.

    WARNING: This does not zoom the image, it only crops or tiles it.

    Args:
        arr (np.ndarray): The input 2D array.
        out_shape (tuple[int, ...]): The target shape. Must have at least 2
            and at most 5 dimensions.
        ensure_unique_info (bool, optional): If True, assumes that `arr` is a label
            image and ensures that labels do not overlap when tiling. Defaults to False.

    Returns:
        np.ndarray: The fitted array with shape `out_shape`.
    """
    if len(out_shape) < 2:
        raise ValueError("`out_shape` must contain at least 2 dimensions.")

    if len(out_shape) > 5:
        raise ValueError("`out_shape` must contain at most 5 dimensions.")

    if any(d <= 0 for d in out_shape):
        raise ValueError("`out_shape` must contain positive integers.")

    if arr.ndim != 2:
        raise ValueError("`arr` must be a 2D array.")

    *_, sy, sx = out_shape
    arr = _fit_to_shape_2d(arr, out_shape=(sy, sx), label_mode=ensure_unique_info)
    arr = np.broadcast_to(arr, out_shape)
    return arr
