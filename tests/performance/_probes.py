"""Counters for work that never reaches the store.

Store counters miss ngio's most expensive habit. `get_ngio_meta` walks a
registry of version decoders and catches `ValidationError` to fall through, so
reading v0.5 metadata costs a *complete failed pydantic validation* before the
right decoder runs — pure CPU, zero store operations. Likewise
`ImageMetaHandler.get_meta` re-parses on every call, and `AbstractImage.
dimensions` calls it on every get and set.

These probes monkeypatch ngio to count that work. They are deliberately brittle:
each patch resolves its target by name and raises `AttributeError` if it has
moved, so a refactor fails loudly instead of silently zeroing a counter and
reading as a performance win.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING

from pydantic import ValidationError
from tests.performance._counting import record

from ngio.hcs._plate import OmeZarrPlate
from ngio.images._abstract_image import AbstractImage
from ngio.ome_zarr_meta import _meta_handlers
from ngio.utils._errors import NgioError
from ngio.utils._zarr_utils import ZarrGroupHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["PROBE_KEYS", "probes"]

#: Counters this module contributes. Kept in sync with `COUNTER_KEYS`.
PROBE_KEYS = (
    "image.dimensions",
    "meta.attrs_load",
    "meta.decode_attempt",
    "meta.decode_call",
    "meta.decode_fail",
    "meta.group_reopen",
    "plate.get_well",
)

_REGISTRY_NAMES = (
    "_image_decoder_registry",
    "_label_decoder_registry",
    "_plate_decoder_registry",
    "_well_decoder_registry",
    "_labels_group_decoder_registry",
)


def _counting_decoder(decoder):
    """Wrap a version decoder so attempts and failures are both counted.

    A failure here is not an error: it is how `get_ngio_meta` decides the attrs
    belong to a different NGFF version. Counting it is the only way to see the
    cost of that guess.
    """

    @wraps(decoder)
    def wrapper(*args, **kwargs):
        record("meta.decode_attempt")
        try:
            return decoder(*args, **kwargs)
        except (ValidationError, NgioError):
            record("meta.decode_fail")
            raise

    return wrapper


def _counting_call(func, counter):
    @wraps(func)
    def wrapper(*args, **kwargs):
        record(counter)
        return func(*args, **kwargs)

    return wrapper


def _counting_property(prop, counter):
    def fget(self):
        record(counter)
        return prop.fget(self)

    return property(fget, prop.fset, prop.fdel, prop.__doc__)


@contextmanager
def probes() -> Iterator[None]:
    """Install the metadata probes for the duration of the block.

    Raises:
        AttributeError: If a patch target no longer exists, which means ngio
            was refactored and the probe must be updated rather than dropped.
    """
    undo = []

    for name in _REGISTRY_NAMES:
        registry = getattr(_meta_handlers, name)
        original = dict(registry)
        undo.append(lambda r=registry, o=original: (r.clear(), r.update(o)))
        for version, decoder in original.items():
            registry[version] = _counting_decoder(decoder)

    for owner, attr, counter in (
        (_meta_handlers, "get_ngio_meta", "meta.decode_call"),
        (ZarrGroupHandler, "reopen_group", "meta.group_reopen"),
        (ZarrGroupHandler, "load_attrs", "meta.attrs_load"),
        (OmeZarrPlate, "_get_well", "plate.get_well"),
    ):
        original_fn = getattr(owner, attr)
        undo.append(lambda o=owner, a=attr, f=original_fn: setattr(o, a, f))
        setattr(owner, attr, _counting_call(original_fn, counter))

    original_dimensions = AbstractImage.__dict__["dimensions"]
    undo.append(lambda: setattr(AbstractImage, "dimensions", original_dimensions))
    AbstractImage.dimensions = _counting_property(
        original_dimensions, "image.dimensions"
    )

    try:
        yield
    finally:
        for restore in reversed(undo):
            restore()
