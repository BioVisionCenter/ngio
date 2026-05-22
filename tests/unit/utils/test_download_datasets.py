from pathlib import Path

import pytest

from ngio.utils import (
    download_ome_zarr_dataset,
    list_ome_zarr_datasets,
    print_datasets_infos,
)


def test_list_datasets():
    assert len(list_ome_zarr_datasets()) > 0
    print_datasets_infos()


def test_fail_download_ome_zarr_dataset(tmp_path: Path):
    tmp_path = Path(tmp_path) / "test_datasets_fail"
    with pytest.raises(ValueError):
        download_ome_zarr_dataset("unknown_dataset", download_dir=tmp_path)

    assert not tmp_path.exists()
