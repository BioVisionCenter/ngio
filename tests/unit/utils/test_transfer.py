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


def test_tables_survive_pack_and_unpack(tmp_path: Path) -> None:
    """Parquet and csv tables are files beside the zarr nodes; copied all the same."""
    from pandas import DataFrame
    from pandas.testing import assert_frame_equal

    from ngio.tables import FeatureTable

    backends = ["anndata", "json", "csv", "parquet"]
    frame = DataFrame({"label": [1, 2, 3], "area": [0.5, 1.5, 2.5]})
    src = _image(tmp_path / "src.zarr")
    for backend in backends:
        src.add_table(
            f"t_{backend}",
            FeatureTable(table_data=frame, reference_label="seg"),
            backend=backend,
        )

    to_ozx(tmp_path / "src.zarr", tmp_path / "a.ozx")
    mem = to_memory(tmp_path / "a.ozx")
    from_memory(mem, tmp_path / "b.zarr")
    from_ozx(tmp_path / "a.ozx", tmp_path / "c.zarr")
    from_memory(mem, tmp_path / "d.ozx")

    for dst in ("a.ozx", "b.zarr", "c.zarr", "d.ozx"):
        img = open_ome_zarr_container(tmp_path / dst, mode="r")
        assert sorted(img.list_tables()) == sorted(f"t_{b}" for b in backends), dst
        for backend in backends:
            got = img.get_table(f"t_{backend}").dataframe.reset_index()
            assert_frame_equal(got.sort_index(axis=1), frame.sort_index(axis=1))
    img = open_ome_zarr_container(mem, mode="r")
    for backend in backends:
        got = img.get_table(f"t_{backend}").dataframe.reset_index()
        assert_frame_equal(got.sort_index(axis=1), frame.sort_index(axis=1))
    assert check_ozx(tmp_path / "a.ozx").must == []
    assert check_ozx(tmp_path / "d.ozx").must == []


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


def test_repack_into_an_archive_held_open_by_path(tmp_path: Path) -> None:
    """`from_memory(..., overwrite=True)` onto an archive a container still holds."""
    held = _image(tmp_path / "held.ozx")
    mem = to_memory(tmp_path / "held.ozx")
    open_ome_zarr_container(mem).derive_label("extra")
    from_memory(mem, tmp_path / "held.ozx", overwrite=True)
    assert (
        "extra"
        in open_ome_zarr_container(tmp_path / "held.ozx", mode="r").list_labels()
    )
    assert "extra" in held.list_labels()
    assert check_ozx(tmp_path / "held.ozx").must == []


def test_copy_onto_itself_is_refused_before_anything_is_cleared(
    tmp_path: Path,
) -> None:
    import gc

    path = tmp_path / "self.ozx"
    held = _image(path)  # live writer in this process
    expected = held.get_image().get_array()
    with pytest.raises(NgioValueError, match="same location"):
        to_ozx(path, path, overwrite=True)
    with pytest.raises(NgioValueError, match="same location"):
        to_ozx(held.get_image().zarr_array.store, path, overwrite=True)
    assert held.list_labels() == ["seg"]  # the caller's container still works
    np.testing.assert_array_equal(held.get_image().get_array(), expected)
    del held
    gc.collect()

    # No live writer: a second handle would truncate the file as it opens.
    with pytest.raises(NgioValueError, match="same location"):
        to_ozx(path, path, overwrite=True)
    with pytest.raises(NgioValueError, match="same location"):
        to_ozx(str(path), tmp_path / "x" / ".." / "self.ozx", overwrite=True)
    assert check_ozx(path).must == []
    np.testing.assert_array_equal(
        open_ome_zarr_container(path, mode="r").get_image().get_array(), expected
    )

    # Directories and memory stores: same guard.
    img_dir = _image(tmp_path / "d.zarr")
    with pytest.raises(NgioValueError, match="same location"):
        to_ozx(tmp_path / "d.zarr", tmp_path / "d.zarr", overwrite=True)
    assert img_dir.list_labels() == ["seg"]
    mem = to_memory(path)
    with pytest.raises(NgioValueError, match="same location"):
        from_memory(mem, mem, overwrite=True)
    _same(open_ome_zarr_container(mem), open_ome_zarr_container(path, mode="r"))


def test_to_memory_from_live_store(tmp_path: Path) -> None:
    from ngio import OzxStore

    with OzxStore(tmp_path / "live.ozx", mode="w") as store:
        img = _image(store)
        mem = to_memory(store)  # a read-only view of the live session
        _same(img, open_ome_zarr_container(mem))
