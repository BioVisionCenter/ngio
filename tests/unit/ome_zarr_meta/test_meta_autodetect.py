"""Guard the version-autodetect trial-decode loop in _meta_handlers.

get_ngio_meta with version=None tries every registered decoder and returns
the first that succeeds. This is only safe while the v0.4 and v0.5 decoders
are mutually exclusive: each must reject the other version's attributes.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from zarr.storage import MemoryStore

from ngio import create_empty_plate, create_ome_zarr_from_array
from ngio.hcs import create_empty_well
from ngio.ome_zarr_meta._meta_handlers import (
    _image_decoder_registry,
    _plate_decoder_registry,
    _well_decoder_registry,
)
from ngio.utils import NgioError, ZarrGroupHandler


def _image_attrs(version: str) -> dict:
    store = MemoryStore()
    create_ome_zarr_from_array(
        store=store,
        array=np.zeros((16, 16), dtype="uint8"),
        pixelsize=1.0,
        axes_names=["y", "x"],
        levels=1,
        ngff_version=version,
    )
    return ZarrGroupHandler(store=store, mode="r").load_attrs()


def _plate_attrs(version: str) -> dict:
    store = MemoryStore()
    create_empty_plate(store=store, name="plate", ngff_version=version)
    return ZarrGroupHandler(store=store, mode="r").load_attrs()


def _well_attrs(version: str) -> dict:
    store = MemoryStore()
    create_empty_well(store=store, ngff_version=version)
    return ZarrGroupHandler(store=store, mode="r").load_attrs()


def _assert_rejected(decoder, attrs, **kwargs):
    """Assert that a decoder refuses the given attrs, in the way the loop expects.

    The exception type *is* part of the contract: the autodetect loop only
    treats `ValidationError` and `NgioError` as "not this version". A decoder
    that rejected with anything else would abort autodetect instead of falling
    through to the next version.
    """
    try:
        decoder(attrs, **kwargs)
    except (ValidationError, NgioError):
        return
    except Exception as e:
        pytest.fail(
            f"decoder rejected with {type(e).__name__}, which the autodetect "
            "loop does not catch; it would abort instead of trying the next "
            "version"
        )
    pytest.fail("decoder unexpectedly accepted attrs of another version")


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_image_decoders_are_version_exclusive(attrs_version):
    attrs = _image_attrs(attrs_version)
    for decoder_version, decoder in _image_decoder_registry.items():
        if decoder_version == attrs_version:
            meta = decoder(attrs, axes_setup=None)
            assert meta.version == attrs_version
        else:
            _assert_rejected(decoder, attrs, axes_setup=None)


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_plate_decoders_are_version_exclusive(attrs_version):
    attrs = _plate_attrs(attrs_version)
    for decoder_version, decoder in _plate_decoder_registry.items():
        if decoder_version == attrs_version:
            assert decoder(attrs).version == attrs_version
        else:
            _assert_rejected(decoder, attrs)


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_well_decoders_are_version_exclusive(attrs_version):
    attrs = _well_attrs(attrs_version)
    for decoder_version, decoder in _well_decoder_registry.items():
        if decoder_version == attrs_version:
            assert decoder(attrs).version == attrs_version
        else:
            _assert_rejected(decoder, attrs)


def test_decoder_bugs_are_not_reported_as_unreadable_metadata(monkeypatch):
    """A crash inside a decoder must surface, not become "failed to decode".

    The loop used to catch bare `Exception`, so any decoder bug looked
    identical to metadata ngio simply cannot read.
    """
    from ngio.ome_zarr_meta import _meta_handlers

    store = MemoryStore()
    create_ome_zarr_from_array(
        store=store,
        array=np.zeros((16, 16), dtype="uint8"),
        pixelsize=1.0,
        axes_names=["y", "x"],
        levels=1,
    )
    handler = ZarrGroupHandler(store=store, mode="r")

    def exploding_decoder(attrs, **kwargs):
        raise AttributeError("bug inside the decoder")

    # Patched only after the store exists, so the crash can only come from the
    # read path under test.
    monkeypatch.setitem(_image_decoder_registry, "0.4", exploding_decoder)
    monkeypatch.setitem(_image_decoder_registry, "0.5", exploding_decoder)

    with pytest.raises(AttributeError, match="bug inside the decoder"):
        _meta_handlers.get_ngio_image_meta(handler)
