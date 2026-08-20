import os
import shutil
from pathlib import Path

import pytest
from filelock import FileLock

from ngio.utils import download_ome_zarr_dataset


def pytest_configure(config):
    """Register custom warning filters."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::ngio.utils._warnings.NgioUserWarning",
    )


ZENODO_DOWNLOAD_DIR = Path(__file__).parent.parent / "data"
TEST_DATA_DIR = Path(__file__).parent / "data"


def _download_dataset(name: str, stamp_dir: Path) -> Path:
    """Download a Zenodo test dataset and re-extract it from its zip.

    Called lazily from session fixtures so that test collection never blocks
    on the network; with a warm cache no network access happens. The datasets
    in `./data` are shared with the docs snippets, which *mutate* them in
    place, so the extracted copy cannot be trusted across runs: the first
    fixture to want a dataset re-extracts it from the pristine zip
    (`re_unzip=True`) and drops a stamp in `stamp_dir` — this pytest run's
    shared temp dir — so later fixtures (and other xdist workers, which would
    otherwise re-extract under a live `copytree`) skip the extraction. The
    file lock serializes the workers around the check.

    Caveat: `filelock`'s Windows backend can hand one lock to two holders, so
    on Windows this does not reliably serialize the workers — see the reason
    text in `ZarrGroupHandler._create_lock`, which is why ngio itself warns
    there. This is a raw `FileLock` rather than ngio's, so it does not even
    carry that warning. Left as is because CI restores `data/` from a cache
    before running, making the cold-cache path it guards rare; a concurrent
    download would corrupt the shared fixtures.
    """
    os.makedirs(ZENODO_DOWNLOAD_DIR, exist_ok=True)
    stamp = stamp_dir / f"{name}.pristine"
    with FileLock(ZENODO_DOWNLOAD_DIR / f"{name}.lock"):
        re_unzip = not stamp.exists()
        path = download_ome_zarr_dataset(
            name, download_dir=ZENODO_DOWNLOAD_DIR, re_unzip=re_unzip
        )
        stamp.touch()
        return path


@pytest.fixture(scope="session")
def _dataset_stamp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory shared by every xdist worker of this pytest run."""
    base = tmp_path_factory.getbasetemp()
    # Under xdist each worker gets `<run dir>/popen-gw*`; the parent is the
    # run dir all workers share. Without xdist, the base is the run dir.
    return base.parent if base.name.startswith("popen-") else base


@pytest.fixture(scope="session")
def cardiomyocyte_tiny_source_path(_dataset_stamp_dir: Path) -> Path:
    return _download_dataset("CardiomyocyteTiny", _dataset_stamp_dir)


@pytest.fixture(scope="session")
def cardiomyocyte_small_mip_source_path(_dataset_stamp_dir: Path) -> Path:
    return _download_dataset("CardiomyocyteSmallMip", _dataset_stamp_dir)


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


@pytest.fixture
def cardiomyocyte_small_mip_image_path(
    tmp_path: Path, cardiomyocyte_small_mip_source_path: Path
) -> Path:
    """A fresh copy of the single image `B/03/0` of the small-mip plate.

    Much cheaper than copying the whole plate; use for tests that write to
    one image only.
    """
    source = cardiomyocyte_small_mip_source_path / "B" / "03" / "0"
    dest_path = tmp_path / "cardiomyocyte_small_mip_image"
    shutil.copytree(source, dest_path)
    return dest_path


@pytest.fixture(scope="session")
def cardiomyocyte_tiny_path_readonly(
    tmp_path_factory: pytest.TempPathFactory, cardiomyocyte_tiny_source_path: Path
) -> Path:
    """One shared session-wide copy of the tiny plate — for read-only tests."""
    dest_path = (
        tmp_path_factory.mktemp("cardio_tiny_ro") / cardiomyocyte_tiny_source_path.stem
    )
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path)
    return dest_path


@pytest.fixture(scope="session")
def cardiomyocyte_small_mip_path_readonly(
    tmp_path_factory: pytest.TempPathFactory, cardiomyocyte_small_mip_source_path: Path
) -> Path:
    """One shared session-wide copy of the small-mip plate — for read-only tests."""
    dest_path = (
        tmp_path_factory.mktemp("cardio_mip_ro")
        / cardiomyocyte_small_mip_source_path.stem
    )
    shutil.copytree(cardiomyocyte_small_mip_source_path, dest_path)
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


def _copy_images_all_versions(dest_base: Path) -> dict[str, Path]:
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


@pytest.fixture
def single_image_copy(tmp_path: Path, zarr_name: str) -> Path:
    """A fresh per-test copy of one test image — for tests that write to it.

    Composes with the `zarr_name` parametrization (or an explicit
    `@pytest.mark.parametrize("zarr_name", ...)` override).
    """
    version, name = zarr_name.split("/")
    source = TEST_DATA_DIR / version / "images" / name
    dest = tmp_path / version / name
    shutil.copytree(source, dest)
    return dest


@pytest.fixture(scope="session")
def images_all_versions_readonly(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """One shared session-wide copy of the test images.

    Use only in tests that never write to the images (a stray write would
    leak into every later test of the session).
    """
    return _copy_images_all_versions(tmp_path_factory.mktemp("images_all_versions"))
