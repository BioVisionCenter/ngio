import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import Roi, create_ome_zarr_from_array
from ngio.iterators._rois_utils import by_grid, by_storage_units
from ngio.utils import NgioValueError


@pytest.fixture
def sample_image():
    container = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros((64, 64), dtype="uint8"),
        pixelsize=1.0,
        axes_names=["y", "x"],
        levels=1,
    )
    return container.get_image()


def full_image_roi() -> Roi:
    return Roi.from_values(
        name="image", slices={"x": (0, 64), "y": (0, 64)}, space="world"
    )


def test_grid_rois_have_unique_names(sample_image):
    tiles = by_grid(
        rois=[full_image_roi()],
        ref_image=sample_image,
        size_x=32,
        size_y=32,
    )
    assert len(tiles) == 4
    names = [tile.name for tile in tiles]
    assert len(set(names)) == len(names)


def test_by_chunks_overlap_equal_to_chunk_size_raises(sample_image):
    chunk_x = sample_image.chunks[-1]
    with pytest.raises(NgioValueError):
        by_storage_units(
            rois=[full_image_roi()],
            ref_image=sample_image,
            overlap_x=chunk_x,
        )
