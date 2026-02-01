import os
import shutil
from pathlib import Path

import pytest

from ngio.utils import download_ome_zarr_dataset

zenodo_download_dir = Path(__file__).parent.parent / "data"
os.makedirs(zenodo_download_dir, exist_ok=True)
cardiomyocyte_tiny_source_path = download_ome_zarr_dataset(
    "CardiomyocyteTiny", download_dir=zenodo_download_dir
)

cardiomyocyte_small_mip_source_path = download_ome_zarr_dataset(
    "CardiomyocyteSmallMip", download_dir=zenodo_download_dir
)


@pytest.fixture
def cardiomyocyte_tiny_path(tmp_path: Path) -> Path:
    dest_path = tmp_path / cardiomyocyte_tiny_source_path.stem
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def cardiomyocyte_small_mip_path(tmp_path: Path) -> Path:
    dest_path = tmp_path / cardiomyocyte_small_mip_source_path.stem
    shutil.copytree(cardiomyocyte_small_mip_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def images_all_versions(tmp_path: Path) -> dict[str, Path]:
    dest_base = tmp_path / "all_versions" / "images"
    dest_base.mkdir(parents=True, exist_ok=True)
    paths = {}
    for version in ["v04", "v05"]:
        source = Path(f"tests/data/{version}/images/")
        dest = dest_base / version
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
        for file in dest.glob("*.zarr"):
            paths[f"{version}/{file.name}"] = file
    return paths
