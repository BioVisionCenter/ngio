"""Iterators to build scalable image processing pipelines."""

from ngio.common._concurrency import MaxWorkers
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder, JobArgs
from ngio.iterators._feature import FeatureExtractorIterator, FeatureFuncResult
from ngio.iterators._image_processing import ImageProcessingIterator
from ngio.iterators._mappers import (
    BasicMapper,
    BatchedMapper,
    IterUnit,
    MapperProtocol,
    ProcessMapper,
    ThreadedMapper,
    canonical_unit_order,
    compute_write_footprint,
    plan_waves,
    write_conflict_components,
)
from ngio.iterators._object_detection import NmsConfig, ObjectDetectionIterator
from ngio.iterators._rois_utils import HaloMargins, TailPolicy
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.iterators._stitch import StitchConfig

__all__ = [
    "AbstractIteratorBuilder",
    "BasicMapper",
    "BatchedMapper",
    "FeatureExtractorIterator",
    "FeatureFuncResult",
    "HaloMargins",
    "ImageProcessingIterator",
    "IterUnit",
    "JobArgs",
    "MapperProtocol",
    "MaskedSegmentationIterator",
    "MaxWorkers",
    "NmsConfig",
    "ObjectDetectionIterator",
    "ProcessMapper",
    "SegmentationIterator",
    "StitchConfig",
    "TailPolicy",
    "ThreadedMapper",
    "canonical_unit_order",
    "compute_write_footprint",
    "plan_waves",
    "write_conflict_components",
]
