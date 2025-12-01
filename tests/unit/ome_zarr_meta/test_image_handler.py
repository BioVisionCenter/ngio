from pathlib import Path

from ngio.ome_zarr_meta import ImageMetaHandler, NgioImageMeta
from ngio.utils import ZarrGroupHandler


def test_get_image_handler(cardiomyocyte_tiny_path: Path):
    # TODO this is a placeholder test
    # The pooch cache is giving us trouble here
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03" / "0"
    group_handler = ZarrGroupHandler(cardiomyocyte_tiny_path)
    handler = ImageMetaHandler(group_handler)
    meta = handler.get_meta()
    assert isinstance(meta, NgioImageMeta)
    handler.update_meta(meta)
