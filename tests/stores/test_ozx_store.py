"""Every table backend on an OME-Zarr zip file, local and remote."""

from __future__ import annotations

import uuid
import zipfile
from typing import TYPE_CHECKING

import fsspec
import pytest
from utils import check_ome_zarr, create_sample_ome_zarr, derive_image

from ngio import check_ozx, open_ome_zarr_container

if TYPE_CHECKING:
    from pathlib import Path

ALL_BACKENDS = ["anndata", "json", "csv", "parquet"]


def _conformant(path: Path) -> None:
    assert check_ozx(path).must == []
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert len(names) == len(set(names))


def test_ozx_store(tmp_path: Path) -> None:
    path = tmp_path / "sample.ozx"
    ome_zarr = create_sample_ome_zarr(
        store=path, supported_backends=ALL_BACKENDS, ngff_version="0.5"
    )
    check_ome_zarr(ome_zarr, supported_backends=ALL_BACKENDS)
    _conformant(path)
    # a fresh handle on the finished file
    check_ome_zarr(
        open_ome_zarr_container(path, mode="r"), supported_backends=ALL_BACKENDS
    )


def test_ozx_store_derive_to_local_and_back(tmp_path: Path) -> None:
    path = tmp_path / "sample.ozx"
    ome_zarr = create_sample_ome_zarr(
        store=path, supported_backends=ALL_BACKENDS, ngff_version="0.5"
    )
    derived = derive_image(ome_zarr, other_store=tmp_path / "derived.zarr")
    check_ome_zarr(derived, supported_backends=ALL_BACKENDS)

    back = derive_image(derived, other_store=tmp_path / "derived.ozx")
    check_ome_zarr(back, supported_backends=ALL_BACKENDS)
    _conformant(tmp_path / "derived.ozx")


@pytest.fixture
def memory_url():
    fs = fsspec.filesystem("memory")
    name = f"tables-{uuid.uuid4().hex}"
    yield f"memory://{name}/sample.ozx"
    fs.rm(f"/{name}", recursive=True)


def test_remote_ozx_tables(tmp_path: Path, memory_url: str) -> None:
    path = tmp_path / "sample.ozx"
    create_sample_ome_zarr(
        store=path, supported_backends=ALL_BACKENDS, ngff_version="0.5"
    )
    fsspec.filesystem("memory").put_file(str(path), memory_url.removeprefix("memory:/"))
    remote = open_ome_zarr_container(memory_url, mode="r")
    check_ome_zarr(remote, supported_backends=ALL_BACKENDS)
