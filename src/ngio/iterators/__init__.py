"""Iterators to build scalable image processing pipelines."""

from ngio.iterators._feature import FeatureExtractorIterator
from ngio.iterators._image_processing import ImageProcessingIterator
from ngio.iterators._mappers import BasicMapper, MapperProtocol
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)

__all__ = [
    "BasicMapper",
    "FeatureExtractorIterator",
    "ImageProcessingIterator",
    "MapperProtocol",
    "MaskedSegmentationIterator",
    "SegmentationIterator",
]
