"""Utilities to build masking regions of interest (ROIs)."""

import itertools
from collections.abc import Sequence

import dask.array as da
import numpy as np
import scipy.ndimage as ndi
from dask.delayed import delayed

from ngio.common._roi import Roi
from ngio.ome_zarr_meta import PixelSize
from ngio.utils import NgioValueError


def _compute_offsets(chunks):
    """Given a chunks tuple, compute cumulative offsets for each axis.

    Returns a list where each element is a list of offsets for that dimension.
    """
    offsets = []
    for dim_chunks in chunks:
        dim_offsets = [0]
        for size in dim_chunks:
            dim_offsets.append(dim_offsets[-1] + size)
        offsets.append(dim_offsets)
    return offsets


def _adjust_slices(slices, offset):
    """Adjust slices to global coordinates using the provided offset."""
    adjusted_slices = {}
    for label, s in slices.items():
        adjusted = tuple(
            slice(s_dim.start + off, s_dim.stop + off)
            for s_dim, off in zip(s, offset, strict=True)
        )
        adjusted_slices[label] = adjusted
    return adjusted_slices


@delayed
def _process_chunk(chunk, offset):
    """Process a single chunk.

    run ndi.find_objects and adjust the slices
    to global coordinates using the provided offset.
    """
    local_slices = compute_slices(chunk)
    local_slices = _adjust_slices(local_slices, offset)
    return local_slices


def _merge_slices(
    slice1: tuple[slice, ...], slice2: tuple[slice, ...]
) -> tuple[slice, ...]:
    """Merge two slices."""
    merged = []
    for s1, s2 in zip(slice1, slice2, strict=True):
        start = min(s1.start, s2.start)
        stop = max(s1.stop, s2.stop)
        merged.append(slice(start, stop))
    return tuple(merged)


@delayed
def _collect_slices(
    local_slices: list[dict[int, tuple[slice, ...]]],
) -> dict[int, tuple[slice]]:
    """Collect the slices from the delayed results."""
    global_slices = {}
    for result in local_slices:
        for label, s in result.items():
            if label in global_slices:
                global_slices[label] = _merge_slices(global_slices[label], s)
            else:
                global_slices[label] = s
    return global_slices


def compute_slices(segmentation: np.ndarray) -> dict[int, tuple[slice, ...]]:
    """Compute slices for each label in a segmentation.

    Args:
        segmentation (ndarray): The segmentation array.

    Returns:
        dict[int, tuple[slice]]: A dictionary with the label as key
            and the slice as value.
    """
    slices = ndi.find_objects(segmentation)
    slices_dict = {}
    for label, s in enumerate(slices, start=1):
        if s is None:
            continue
        else:
            slices_dict[label] = s
    return slices_dict


def lazy_compute_slices(segmentation: da.Array) -> dict[int, tuple[slice, ...]]:
    """Compute slices for each label in a segmentation."""
    global_offsets = _compute_offsets(segmentation.chunks)
    delayed_chunks = segmentation.to_delayed()  # type: ignore

    grid_shape = tuple(len(c) for c in segmentation.chunks)

    grid_indices = list(itertools.product(*[range(n) for n in grid_shape]))
    delayed_results = []
    for idx, chunk in zip(grid_indices, np.ravel(delayed_chunks), strict=True):
        offset = tuple(global_offsets[dim][idx[dim]] for dim in range(len(idx)))
        delayed_result = _process_chunk(chunk, offset)
        delayed_results.append(delayed_result)

    return _collect_slices(delayed_results).compute()


def compute_masking_roi(
    segmentation: np.ndarray | da.Array,
    pixel_size: PixelSize,
    axes_order: Sequence[str],
) -> list[Roi]:
    """Compute a ROIs for each label in a segmentation.

    This function expects a 2D or 3D segmentation array.
    And this function expects the axes order to be 'zyx' or 'yx'.
    Other axes orders are not supported.

    """
    if segmentation.ndim not in [2, 3, 4]:
        raise NgioValueError("Only 2D, 3D, and 4D segmentations are supported.")

    if len(axes_order) != segmentation.ndim:
        raise NgioValueError(
            "The length of axes_order must match the number of dimensions "
            "of the segmentation."
        )

    if isinstance(segmentation, da.Array):
        slices = lazy_compute_slices(segmentation)
    else:
        slices = compute_slices(segmentation)

    rois = []
    for label, slice_ in slices.items():
        assert len(slice_) == len(axes_order)
        slices = dict(zip(axes_order, slice_, strict=True))
        roi = Roi.from_values(
            name=str(label), slices=slices, label=label, space="pixel"
        )
        roi = roi.to_world(pixel_size=pixel_size)
        rois.append(roi)
    return rois
