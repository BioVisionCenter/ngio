"""`to_memory` / `from_memory` / `to_ozx` / `from_ozx`."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import (
    check_ozx,
    create_empty_ome_zarr,
    from_memory,
    from_ozx,
    open_ome_zarr_container,
    to_memory,
    to_ozx,
)
from ngio.utils import NgioFileExistsError, NgioValueError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SHAPE = (2, 4, 16, 16)


def _image(store, version="0.5"):
    img = create_empty_ome_zarr(
        store, shape=SHAPE, pixelsize=0.5, chunks=(1, 2, 8, 8), ngff_version=version
    )
    img.get_image().set_array(np.random.randint(0, 255, size=SHAPE, dtype=np.uint8))
    img.derive_label("seg").set_array(np.ones(SHAPE[1:], dtype=np.uint8))
    img.set_channel_labels(["a", "b"])
    return img


def _same(a, b) -> None:
    assert a.list_labels() == b.list_labels() == ["seg"]
    assert a.get_image().channel_labels == b.get_image().channel_labels
    np.testing.assert_array_equal(a.get_image().get_array(), b.get_image().get_array())
    np.testing.assert_array_equal(
        a.get_label("seg").get_array(), b.get_label("seg").get_array()
    )


def test_dir_to_ozx_to_memory_to_dir(tmp_path: Path) -> None:
    src = _image(tmp_path / "src.zarr")
    ozx = to_ozx(tmp_path / "src.zarr", tmp_path / "packed.ozx")
    report = check_ozx(ozx)
    assert report.conforming and report.should == [
        "ZIP64 end of central directory records are not present"
    ]
    _same(src, open_ome_zarr_container(ozx))

    mem = to_memory(ozx)
    assert isinstance(mem, MemoryStore)
    in_memory = open_ome_zarr_container(mem)
    _same(src, in_memory)
    in_memory.set_channel_labels(["x", "y"])  # edits stay in RAM
    assert open_ome_zarr_container(ozx).get_image().channel_labels == ["a", "b"]

    from_memory(mem, tmp_path / "back.zarr")
    assert open_ome_zarr_container(
        tmp_path / "back.zarr"
    ).get_image().channel_labels == ["x", "y"]
    from_memory(mem, ozx, overwrite=True)  # repack the edited copy over the archive
    assert check_ozx(ozx).conforming
    assert open_ome_zarr_container(ozx).get_image().channel_labels == ["x", "y"]

    from_ozx(ozx, tmp_path / "unpacked.zarr")
    _same(in_memory, open_ome_zarr_container(tmp_path / "unpacked.zarr"))


def test_overwrite_flag(tmp_path: Path) -> None:
    _image(tmp_path / "src.zarr")
    to_ozx(tmp_path / "src.zarr", tmp_path / "p.ozx")
    with pytest.raises(NgioFileExistsError):
        to_ozx(tmp_path / "src.zarr", tmp_path / "p.ozx")
    to_ozx(tmp_path / "src.zarr", tmp_path / "p.ozx", overwrite=True)
    from_ozx(tmp_path / "p.ozx", tmp_path / "out.zarr")
    with pytest.raises(NgioFileExistsError):
        from_ozx(tmp_path / "p.ozx", tmp_path / "out.zarr")
    from_ozx(tmp_path / "p.ozx", tmp_path / "out.zarr", overwrite=True)


def test_v2_source_cannot_be_packed(tmp_path: Path) -> None:
    _image(tmp_path / "v04.zarr", version="0.4")
    with pytest.raises(NgioValueError, match="Zarr v2"):
        to_ozx(tmp_path / "v04.zarr", tmp_path / "v04.ozx")
    assert not (tmp_path / "v04.ozx").exists()


@pytest.fixture
def memory_url() -> Iterator[str]:
    """A URL on fsspec's in-process `memory://` filesystem; removed afterwards."""
    fs = fsspec.filesystem("memory")
    name = f"remote-{uuid.uuid4().hex}"
    yield f"memory://{name}/image.ozx"
    fs.rm(f"/{name}", recursive=True)


def test_remote_ozx_round_trip(tmp_path: Path, memory_url: str) -> None:
    src = _image(tmp_path / "src.zarr")
    assert to_ozx(tmp_path / "src.zarr", memory_url) == memory_url
    assert check_ozx(memory_url).must == []

    # read-only in place, over the (memory) filesystem
    _same(src, open_ome_zarr_container(memory_url, mode="r"))
    with pytest.raises(NgioValueError, match="read-only"):
        open_ome_zarr_container(memory_url, mode="r+")

    from_ozx(memory_url, tmp_path / "back.zarr")
    _same(src, open_ome_zarr_container(tmp_path / "back.zarr", mode="r"))

    with pytest.raises(NgioFileExistsError):
        to_ozx(tmp_path / "src.zarr", memory_url)
    from_memory(to_memory(tmp_path / "src.zarr"), memory_url, overwrite=True)
    _same(src, open_ome_zarr_container(memory_url, mode="r"))


def test_to_memory_from_live_store(tmp_path: Path) -> None:
    from ngio import OzxStore

    with OzxStore(tmp_path / "live.ozx", mode="w") as store:
        img = _image(store)
        mem = to_memory(store)  # a read-only view of the live session
        _same(img, open_ome_zarr_container(mem))
