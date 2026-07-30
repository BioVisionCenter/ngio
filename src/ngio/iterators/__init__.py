"""Iterators to build scalable image processing pipelines."""

from ngio.iterators._feature import FeatureExtractorIterator
from ngio.iterators._image_processing import ImageProcessingIterator
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)

__all__ = [
    "FeatureExtractorIterator",
    "ImageProcessingIterator",
    "MaskedSegmentationIterator",
    "SegmentationIterator",
]
