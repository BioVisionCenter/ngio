"""This file is part of NGIO, a library for working with OME-Zarr data."""

from ngio.experimental.iterators._feature import FeatureExtractorIterator
from ngio.experimental.iterators._image_processing import ImageProcessingIterator
from ngio.experimental.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)

# from ngio.experimental.iterators._builder import IteratorBuilder

__all__ = [
    "FeatureExtractorIterator",
    "ImageProcessingIterator",
    "MaskedSegmentationIterator",
    "SegmentationIterator",
]
