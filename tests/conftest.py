import os
import shutil
from pathlib import Path

import pytest

from ngio.utils import download_ome_zarr_dataset


def pytest_configure(config):
    """Register custom warning filters."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::ngio.utils._warnings.NgioUserWarning",
    )


ZENODO_DOWNLOAD_DIR = Path(__file__).parent.parent / "data"
TEST_DATA_DIR = Path(__file__).parent / "data"


def _download_dataset(name: str) -> Path:
    """Download (or reuse the cached copy of) a Zenodo test dataset.

    Called lazily from session fixtures so that test collection never
    blocks on the network; with a warm cache no network access happens.
    """
    os.makedirs(ZENODO_DOWNLOAD_DIR, exist_ok=True)
    return download_ome_zarr_dataset(name, download_dir=ZENODO_DOWNLOAD_DIR)


@pytest.fixture(scope="session")
def cardiomyocyte_tiny_source_path() -> Path:
    return _download_dataset("CardiomyocyteTiny")


@pytest.fixture(scope="session")
def cardiomyocyte_small_mip_source_path() -> Path:
    return _download_dataset("CardiomyocyteSmallMip")


@pytest.fixture
def cardiomyocyte_tiny_path(
    tmp_path: Path, cardiomyocyte_tiny_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_tiny_source_path.stem
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def cardiomyocyte_small_mip_path(
    tmp_path: Path, cardiomyocyte_small_mip_source_path: Path
) -> Path:
    dest_path = tmp_path / cardiomyocyte_small_mip_source_path.stem
    shutil.copytree(cardiomyocyte_small_mip_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


# One entry per (NGFF version, axes combination) in tests/data/{v04,v05}/images
ALL_IMAGE_ZARR_NAMES = [
    f"{version}/test_image_{axes}.zarr"
    for version in ("v04", "v05")
    for axes in ("yx", "cyx", "zyx", "czyx", "c1yx", "tyx", "tcyx", "tzyx", "tczyx")
]


@pytest.fixture(params=ALL_IMAGE_ZARR_NAMES)
def zarr_name(request: pytest.FixtureRequest) -> str:
    """Name of one on-disk test image, keyed into `images_all_versions`."""
    return request.param


@pytest.fixture
def images_all_versions(tmp_path: Path) -> dict[str, Path]:
    dest_base = tmp_path / "all_versions" / "images"
    dest_base.mkdir(parents=True, exist_ok=True)
    paths = {}
    for version in ["v04", "v05"]:
        source = TEST_DATA_DIR / version / "images"
        dest = dest_base / version
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
        for file in dest.glob("*.zarr"):
            paths[f"{version}/{file.name}"] = file
    return paths
