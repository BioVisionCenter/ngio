"""This file is part of NGIO, a library for working with OME-Zarr data."""

from ngio.experimental.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)

# from ngio.experimental.iterators._builder import IteratorBuilder

__all__ = [
    "MaskedSegmentationIterator",
    "SegmentationIterator",
]
