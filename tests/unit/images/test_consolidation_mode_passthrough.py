"""The consolidation-mode deprecation notice must always be actionable.

`_resolve_mode` warns when `mode is None` and the coming `"auto"` default
would change the result. Every ngio entry point that consolidates internally
must either pin a mode or expose a passthrough — the warning must never tell
a user to pass a kwarg that the function they actually called does not have.
"""

import warnings

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array, create_synthetic_ome_zarr
from ngio.iterators import ImageProcessingIterator, SegmentationIterator
from ngio.utils import NgioFutureWarning

# Small and dyadic, so `auto` would resolve to numpy and the notice is due.
_ARRAY = np.zeros((64, 64), dtype="uint16")


def test_create_from_array_default_warns_and_is_actionable():
    with pytest.warns(NgioFutureWarning, match="consolidation_mode"):
        create_ome_zarr_from_array(store=MemoryStore(), array=_ARRAY, pixelsize=1.0)


@pytest.mark.parametrize("mode", ["dask", "auto"])
def test_create_from_array_explicit_mode_is_silent(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=NgioFutureWarning)
        create_ome_zarr_from_array(
            store=MemoryStore(), array=_ARRAY, pixelsize=1.0, consolidation_mode=mode
        )


def test_create_synthetic_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=NgioFutureWarning)
        create_synthetic_ome_zarr(store=MemoryStore(), shape=(64, 64))


def _container_with_label():
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=_ARRAY, pixelsize=1.0, consolidation_mode="dask"
    )
    label = ome_zarr.derive_label("label")
    return ome_zarr, label


def test_segmentation_iterator_consolidation_mode_passthrough():
    ome_zarr, label = _container_with_label()
    image = ome_zarr.get_image()

    with pytest.warns(NgioFutureWarning, match="consolidation_mode"):
        SegmentationIterator(image, label).finalize()

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=NgioFutureWarning)
        iterator = SegmentationIterator(image, label, consolidation_mode="dask")
        iterator.finalize()
        # The mode survives the rebuilds `by_yx`/`by_chunks` perform.
        iterator.by_yx().finalize()


def test_image_processing_iterator_consolidation_mode_passthrough():
    ome_zarr, _ = _container_with_label()
    image = ome_zarr.get_image()
    out = ome_zarr.derive_image(store=MemoryStore()).get_image()

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=NgioFutureWarning)
        iterator = ImageProcessingIterator(image, out, consolidation_mode="dask")
        iterator.finalize()
        iterator.by_yx().finalize()
