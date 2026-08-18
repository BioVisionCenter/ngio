"""Transforms and merge policies for the io pipes.

Two different things live here, and the split is the point.

A **transform** is a function of the patch alone. The read chain applies it,
the write chain inverts it, and a list of them composes in either direction
with no ordering rules — `transforms=[...]`.

A **merge policy** is a function of the patch *and* what is already on disk:
how the two combine. It runs once, after the transform chain, in the array's
own space — `merge=...` on the write.
"""

from ngio.io_pipes import (
    AxesOps,
    IoPipeContext,
    SlicingOps,
    TransformContext,
    TransformProtocol,
)
from ngio.io_pipes._merge_policy import MergePolicy, MergeRule
from ngio.transforms._mask import MaskMerge, MaskTransform
from ngio.transforms._unique_labels import UniqueLabelsTransform
from ngio.transforms._zoom import ZoomTransform

__all__ = [
    "AxesOps",
    "IoPipeContext",
    "MaskMerge",
    "MaskTransform",
    "MergePolicy",
    "MergeRule",
    "SlicingOps",
    "TransformContext",
    "TransformProtocol",
    "UniqueLabelsTransform",
    "ZoomTransform",
]
