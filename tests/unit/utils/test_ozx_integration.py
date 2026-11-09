"""OME-Zarr zip files through the public ngio API (containers, plates, paths)."""

from __future__ import annotations

import gc
import pickle
import subprocess
import sys
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ngio import (
    ImageInWellPath,
    OzxStore,
    check_ozx,
    create_empty_ome_zarr,
    create_empty_plate,
    create_ome_zarr_from_array,
    open_ome_zarr_container,
    open_ome_zarr_plate,
)
from ngio.utils import NgioValueError

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (2, 4, 16, 16)
CHUNKS = (1, 2, 8, 8)


def _assert_conforming(path: Path) -> None:
    report = check_ozx(path)
    assert report.conforming, report
    assert report.json_first and not report.duplicates, report
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert names[0] == "zarr.json"


def _create(store, **kwargs):
    return create_empty_ome_zarr(
        store, shape=SHAPE, pixelsize=0.5, chunks=CHUNKS, ngff_version="0.5", **kwargs
    )


class TestExplicitStore:
    """The `with OzxStore(...)` form: the session end is explicit."""

    def test_write_then_read(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        data = np.random.randint(0, 255, size=SHAPE, dtype=np.uint8)
        with OzxStore(path, mode="w") as store:
            img = _create(store)
            _assert_conforming(path)  # readable as soon as the image exists
            img.get_image().set_array(data)
            img.set_space_unit("nanometer")
            img.set_channel_labels(["a", "b"])
            img.set_channel_labels(["c", "d"])
            _assert_conforming(path)
            label = img.derive_label("seg")
            label.set_array(np.ones(SHAPE[1:], dtype=np.uint16))
        _assert_conforming(path)
        ro = open_ome_zarr_container(path)
        assert ro.list_labels() == ["seg"]
        np.testing.assert_array_equal(ro.get_image().get_array(), data)
        assert ro.get_image().channel_labels == ["c", "d"]
        assert ro.get_image().space_unit == "nanometer"

    def test_second_session_overwrite_delete_repack(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            img = _create(store)
            img.derive_label("a").set_array(np.ones(SHAPE[1:], dtype=np.uint8))
            img.derive_label("b")
        size_1 = path.stat().st_size

        with OzxStore(path, mode="r+") as store:
            img = open_ome_zarr_container(store)
            img.derive_label("a", overwrite=True)
            img.delete_label("b")
            img.set_channel_labels(["x", "y"])
            _assert_conforming(path)
            assert img.list_labels() == ["a"]
            assert path.stat().st_size > size_1
            store.repack()
            assert path.stat().st_size < size_1
            img.get_image().set_array(np.full(SHAPE, 7, dtype=np.uint8))
        _assert_conforming(path)
        ro = open_ome_zarr_container(path)
        assert ro.list_labels() == ["a"]
        assert ro.get_image().channel_labels == ["x", "y"]
        assert int(ro.get_image().get_array()[0, 0, 0, 0]) == 7

    def test_create_from_array(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        data = np.random.randint(0, 255, size=SHAPE, dtype=np.uint8)
        with OzxStore(path, mode="w") as store:
            img = create_ome_zarr_from_array(
                store, data, pixelsize=0.5, chunks=CHUNKS, ngff_version="0.5"
            )
            assert img.levels == 5
        _assert_conforming(path)
        np.testing.assert_array_equal(
            open_ome_zarr_container(path).get_image().get_array(), data
        )

    def test_plate(self, tmp_path: Path) -> None:
        path = tmp_path / "plate.ozx"
        with OzxStore(path, mode="w") as store:
            plate = create_empty_plate(
                store,
                name="p",
                images=[ImageInWellPath(row="A", column="1", path="0")],
                ngff_version="0.5",
            )
            plate.add_image("B", "2", "0")
            _assert_conforming(path)
        plate = open_ome_zarr_plate(path)
        assert plate.rows == ["A", "B"]
        assert plate.columns == ["01", "02"]

    def test_plate_from_path(self, tmp_path: Path) -> None:
        path = tmp_path / "plate.ozx"
        plate = create_empty_plate(path, name="p", ngff_version="0.5")
        plate.add_image("A", "1", "0")
        del plate
        gc.collect()
        _assert_conforming(path)
        assert open_ome_zarr_plate(path).rows == ["A"]


class TestPathRouting:
    """A generic script passes a path and never learns it was a zip."""

    def test_writable_path_finalized_on_release(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        img = _create(path, overwrite=True)
        img.get_image().set_array(np.ones(SHAPE, dtype=np.uint8))
        img.set_channel_labels(["a", "b"])
        _assert_conforming(path)  # the metadata write published the hierarchy
        del img
        gc.collect()
        _assert_conforming(path)
        ro = open_ome_zarr_container(path, mode="r")
        assert int(ro.get_image().get_array().max()) == 1

        img = open_ome_zarr_container(path, mode="r+")
        img.derive_label("seg")
        del img
        gc.collect()
        assert open_ome_zarr_container(path).list_labels() == ["seg"]

    def test_chunk_writes_publish_before_release(self, tmp_path: Path) -> None:
        # Random bytes: compressible data hides a missing directory behind
        # the write buffer. Everything stays referenced throughout.
        path = tmp_path / "img.ozx"
        rng = np.random.default_rng(0)
        data = rng.integers(0, 255, size=SHAPE, dtype=np.uint8)
        oz = _create(path, overwrite=True)
        img = oz.get_image()
        img.set_array(data)
        _assert_conforming(path)
        other = open_ome_zarr_container(path, mode="r").get_image()
        np.testing.assert_array_equal(other.get_array(), data)

        patch = rng.integers(0, 255, size=(2, 4, 8, 8), dtype=np.uint8)
        img.set_roi(img.roi(), patch, x=slice(0, 8), y=slice(0, 8))
        _assert_conforming(path)
        img.consolidate()
        _assert_conforming(path)
        with zipfile.ZipFile(path) as zf:
            assert any(name.startswith("1/c/") for name in zf.namelist())

    def test_chunk_writes_survive_os_exit(self, tmp_path: Path) -> None:
        # `os._exit` skips finalizers and garbage collection: what is on
        # disk when `set_array` returns is all a crashed process leaves.
        path = tmp_path / "img.ozx"
        script = (
            "import os\nimport numpy as np\n"
            "from ngio import create_empty_ome_zarr\n"
            f"oz = create_empty_ome_zarr({str(path)!r}, shape={SHAPE!r}, "
            f"pixelsize=0.5, chunks={CHUNKS!r}, ngff_version='0.5')\n"
            "rng = np.random.default_rng(0)\n"
            f"oz.get_image().set_array(rng.integers(0, 255, size={SHAPE!r}, "
            "dtype=np.uint8))\n"
            "os._exit(0)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        _assert_conforming(path)
        expected = np.random.default_rng(0).integers(0, 255, size=SHAPE, dtype=np.uint8)
        img = open_ome_zarr_container(path, mode="r").get_image()
        np.testing.assert_array_equal(img.get_array(), expected)

    def test_mode_w_truncates(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _create(store).derive_label("old")
        img = _create(path, overwrite=True)
        del img
        gc.collect()
        assert open_ome_zarr_container(path).list_labels() == []

    def test_zarr_v2_refused(self, tmp_path: Path) -> None:
        with pytest.raises(NgioValueError, match="Zarr v2"):
            create_empty_ome_zarr(
                tmp_path / "img.ozx", shape=SHAPE, pixelsize=0.5, ngff_version="0.4"
            )

    def test_one_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _create(store)
            with pytest.raises(NgioValueError, match="already open for writing"):
                # the archive is opened lazily, on first IO
                open_ome_zarr_container(OzxStore(path, mode="r+"))
            # By path, generic code joins the session instead of failing.
            joined = open_ome_zarr_container(path, mode="r+")
            joined.set_channel_labels(["a", "b"])
            assert open_ome_zarr_container(store).get_image().channel_labels == [
                "a",
                "b",
            ]
        with pytest.raises(NgioValueError, match="closed"):
            joined.get_image().get_array()

    def test_reopen_by_path_shares_the_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        created = _create(path)  # holds the writable store
        edited = open_ome_zarr_container(path)  # default mode r+
        edited.set_channel_labels(["x", "y"])
        assert created.get_image().channel_labels == ["x", "y"]
        assert open_ome_zarr_container(path, mode="r").get_image().channel_labels == [
            "x",
            "y",
        ]
        # mode="w" empties the live archive in place rather than opening a 2nd handle
        fresh = _create(path, overwrite=True)
        assert fresh.list_labels() == []
        _assert_conforming(path)

    def test_remote_path_is_read_only(self) -> None:
        with pytest.raises(NgioValueError, match="read-only"):
            open_ome_zarr_container("https://example.org/x.ozx")  # default r+


@pytest.mark.network
def test_public_remote_ozx() -> None:
    url = "https://static.webknossos.org/misc/6001240.ozx"
    assert check_ozx(url).must == []
    img = open_ome_zarr_container(url, mode="r").get_image()
    assert img.get_array(x=slice(0, 4), y=slice(0, 4)).size > 0

    def test_pickle_read_only_container_store(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _create(store).get_image().set_array(np.ones(SHAPE, dtype=np.uint8))
        ro = open_ome_zarr_container(path, mode="r")
        store = ro._group_handler.store
        copy = pickle.loads(pickle.dumps(store))
        assert copy.is_ozx
        img = open_ome_zarr_container(path, mode="r+")
        with pytest.raises(NgioValueError, match="cannot be pickled"):
            pickle.dumps(img._group_handler.store)
