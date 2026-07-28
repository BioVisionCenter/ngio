"""Guard the version-autodetect trial-decode loop in _meta_handlers.

get_ngio_meta with version=None tries every registered decoder and returns
the first that succeeds. This is only safe while the v0.4 and v0.5 decoders
are mutually exclusive: each must reject the other version's attributes.
"""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_empty_plate, create_ome_zarr_from_array
from ngio.hcs import create_empty_well
from ngio.ome_zarr_meta._meta_handlers import (
    _image_decoder_registry,
    _plate_decoder_registry,
    _well_decoder_registry,
)
from ngio.utils import ZarrGroupHandler


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


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_image_decoders_are_version_exclusive(attrs_version):
    attrs = _image_attrs(attrs_version)
    for decoder_version, decoder in _image_decoder_registry.items():
        if decoder_version == attrs_version:
            meta = decoder(attrs, axes_setup=None)
            assert meta.version == attrs_version
        else:
            with pytest.raises(Exception):
                decoder(attrs, axes_setup=None)


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_plate_decoders_are_version_exclusive(attrs_version):
    attrs = _plate_attrs(attrs_version)
    for decoder_version, decoder in _plate_decoder_registry.items():
        if decoder_version == attrs_version:
            assert decoder(attrs).version == attrs_version
        else:
            with pytest.raises(Exception):
                decoder(attrs)


@pytest.mark.parametrize("attrs_version", ["0.4", "0.5"])
def test_well_decoders_are_version_exclusive(attrs_version):
    attrs = _well_attrs(attrs_version)
    for decoder_version, decoder in _well_decoder_registry.items():
        if decoder_version == attrs_version:
            assert decoder(attrs).version == attrs_version
        else:
            with pytest.raises(Exception):
                decoder(attrs)
