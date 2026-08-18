"""Iterators to build scalable image processing pipelines."""

from ngio.iterators._feature import FeatureExtractorIterator
from ngio.iterators._image_processing import ImageProcessingIterator
from ngio.iterators._mappers import (
    BasicMapper,
    IterUnit,
    MapperProtocol,
    ProcessMapper,
    ThreadedMapper,
)
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.iterators._stitch import StitchConfig

__all__ = [
    "BasicMapper",
    "FeatureExtractorIterator",
    "ImageProcessingIterator",
    "IterUnit",
    "MapperProtocol",
    "MaskedSegmentationIterator",
    "ProcessMapper",
    "SegmentationIterator",
    "StitchConfig",
    "ThreadedMapper",
]
