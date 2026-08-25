"""Iterators to build scalable image processing pipelines."""

from ngio.common._concurrency import MaxWorkers
from ngio.iterators._abstract_iterator import (
    AbstractIteratorBuilder,
    JobArgs,
    OverlapPolicy,
)
from ngio.iterators._feature import (
    ConcatJoin,
    FeatureExtractorIterator,
    FeatureFuncResult,
    JoinProtocol,
)
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
from ngio.iterators._object_detection import (
    Detection,
    GreedyNms,
    NmsProtocol,
    ObjectDetectionIterator,
    bbox_iou,
)
from ngio.iterators._rois_utils import HaloMargins, TailPolicy
from ngio.iterators._segmentation import (
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.iterators._stitch import IouSeamMatcher, SeamMatcherProtocol, StitchConfig

__all__ = [
    "AbstractIteratorBuilder",
    "BasicMapper",
    "BatchedMapper",
    "ConcatJoin",
    "Detection",
    "FeatureExtractorIterator",
    "FeatureFuncResult",
    "GreedyNms",
    "HaloMargins",
    "ImageProcessingIterator",
    "IouSeamMatcher",
    "IterUnit",
    "JobArgs",
    "JoinProtocol",
    "MapperProtocol",
    "MaskedSegmentationIterator",
    "MaxWorkers",
    "NmsProtocol",
    "ObjectDetectionIterator",
    "OverlapPolicy",
    "ProcessMapper",
    "SeamMatcherProtocol",
    "SegmentationIterator",
    "StitchConfig",
    "TailPolicy",
    "ThreadedMapper",
    "bbox_iou",
    "canonical_unit_order",
    "compute_write_footprint",
    "plan_waves",
    "write_conflict_components",
]
