"""Iterators to build scalable image processing pipelines."""

from ngio.common._concurrency import MaxWorkers
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.iterators._feature import FeatureExtractorIterator, FeatureFuncResult
from ngio.iterators._image_processing import ImageProcessingIterator
from ngio.iterators._mappers import (
    BasicMapper,
    IterUnit,
    MapperProtocol,
    ProcessMapper,
    ThreadedMapper,
    compute_write_footprint,
    write_conflict_components,
)
from ngio.iterators._object_detection import (
    DetectionFuncResult,
    NmsConfig,
    ObjectDetectionIterator,
)
from ngio.iterators._rois_utils import HaloMargins
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.iterators._stitch import StitchConfig

__all__ = [
    "AbstractIteratorBuilder",
    "BasicMapper",
    "DetectionFuncResult",
    "FeatureExtractorIterator",
    "FeatureFuncResult",
    "HaloMargins",
    "ImageProcessingIterator",
    "IterUnit",
    "MapperProtocol",
    "MaskedSegmentationIterator",
    "MaxWorkers",
    "NmsConfig",
    "ObjectDetectionIterator",
    "ProcessMapper",
    "SegmentationIterator",
    "StitchConfig",
    "ThreadedMapper",
    "compute_write_footprint",
    "write_conflict_components",
]
