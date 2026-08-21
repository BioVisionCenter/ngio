"""Deprecated alias for `ngio.iterators`.

The iterators are no longer experimental. Import them from `ngio.iterators`
(or from `ngio` directly). This module will be removed in `ngio=1.2`.
"""

import warnings
from typing import TYPE_CHECKING, Any

from ngio.utils import NgioDeprecationWarning

if TYPE_CHECKING:
    from ngio.iterators import (
        FeatureExtractorIterator,
        ImageProcessingIterator,
        MaskedSegmentationIterator,
        SegmentationIterator,
    )

__all__ = [
    "FeatureExtractorIterator",
    "ImageProcessingIterator",
    "MaskedSegmentationIterator",
    "SegmentationIterator",
]


def __getattr__(name: str) -> Any:
    """Forward attribute access to `ngio.iterators` with a deprecation warning."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"'ngio.experimental.iterators.{name}' is deprecated and will be removed "
        f"in ngio=1.2. Please use 'ngio.iterators.{name}' instead.",
        NgioDeprecationWarning,
        stacklevel=2,
    )
    import ngio.iterators

    return getattr(ngio.iterators, name)
