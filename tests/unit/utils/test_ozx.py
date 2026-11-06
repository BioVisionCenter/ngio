"""`OzxStore`: an RFC-9 OME-Zarr zip file as a zarr store (store level only)."""

from __future__ import annotations

import gc
import pickle
import subprocess
import sys
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ngio.utils._errors import NgioValueError
from ngio.utils._ozx import OzxStore, _OzxZipFile, check_ozx, rfc9_comment
from ngio.utils._warnings import NgioUserWarning

if TYPE_CHECKING:
    from pathlib import Path

ROOT_ATTRS = {"ome": {"version": "0.5"}}


def _names(path: Path) -> list[str]:
    """What a reader with its own handle sees, after an integrity check."""
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None
        return zf.namelist()


def _assert_conforming(path: Path) -> None:
    report = check_ozx(path)
    assert report.conforming, report
    assert report.json_first and not report.duplicates, report
    names = _names(path)
    assert len(names) == len(set(names))
    assert names[0] == "zarr.json"


def _write_hierarchy(store: OzxStore) -> np.ndarray:
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs.put(ROOT_ATTRS)
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)
    arr = root.create_array("0", shape=data.shape, chunks=(4, 4), dtype=data.dtype)
    arr[:] = data
    labels = root.create_group("labels")
    labels.create_array("a", shape=(2,), chunks=(2,), dtype="u1")[:] = [1, 2]
    return data


class TestSession:
    def test_write_rewrite_close(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            data = _write_hierarchy(store)
            root = zarr.open_group(store, mode="r+")
            # Rewriting metadata leaves one record, not a duplicate.
            root.attrs["ome"] = {"version": "0.5", "name": "x"}
            root.attrs["ome"] = {"version": "0.5", "name": "y"}
            store.flush()
            assert _names(path).count("zarr.json") == 1
            _assert_conforming(path)
        assert store.closed
        _assert_conforming(path)
        with zipfile.ZipFile(path) as zf:
            assert zf.comment == rfc9_comment("0.5")
            # root first, then breadth-first, then everything else
            names = zf.namelist()
            assert names[:4] == [
                "zarr.json",
                "0/zarr.json",
                "labels/zarr.json",
                "labels/a/zarr.json",
            ]
        root = zarr.open_group(OzxStore(path, mode="r"), mode="r")
        assert root.attrs["ome"]["name"] == "y"
        np.testing.assert_array_equal(root["0"][:], data)

    def test_append_session_delete_overwrite_repack(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _write_hierarchy(store)
        size_1 = path.stat().st_size

        with OzxStore(path, mode="r+") as store:
            root = zarr.open_group(store, mode="r+")
            del root["labels"]
            assert "labels" not in root
            assert sync(store.exists("labels/zarr.json")) is False
            arr = root.create_array(
                "0", shape=(8, 8), chunks=(4, 4), dtype="u1", overwrite=True
            )
            arr[:] = 7
            store.flush()
            _assert_conforming(path)
            assert path.stat().st_size > size_1  # dead space
            store.repack()
            assert path.stat().st_size < size_1
            _assert_conforming(path)
            assert check_ozx(path).should == [
                "ZIP64 end of central directory records are not present"
            ]
            arr[0, 0] = 9  # the session continues after a repack
        _assert_conforming(path)
        root = zarr.open_group(OzxStore(path, mode="r"), mode="r")
        assert "labels" not in root
        assert int(root["0"][0, 0]) == 9 and int(root["0"][1, 1]) == 7

    def test_untouched_session_leaves_file_identical(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _write_hierarchy(store)
        before = path.read_bytes()
        with OzxStore(path, mode="r+") as store:
            zarr.open_group(store, mode="r+")
        assert path.read_bytes() == before

    def test_mode_w_truncates_and_w_minus_is_exclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w-") as store:
            _write_hierarchy(store)
        with pytest.raises(FileExistsError):
            zarr.open_group(OzxStore(path, mode="w-"), mode="w-")
        with OzxStore(path, mode="w") as store:
            zarr.open_group(store, mode="w", zarr_format=3).attrs.put(ROOT_ATTRS)
        assert _names(path) == ["zarr.json"]

    def test_threaded_dask_write(self, tmp_path: Path) -> None:
        da = pytest.importorskip("dask.array")
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            root = zarr.open_group(store, mode="w", zarr_format=3)
            root.attrs.put(ROOT_ATTRS)
            arr = root.create_array("0", shape=(16, 16), chunks=(4, 4), dtype="u1")
            source = da.random.randint(
                0, 255, size=(16, 16), dtype=np.uint8, chunks=(4, 4)
            )
            da.store(source, arr, lock=False)
            expected = source.compute()
        _assert_conforming(path)
        np.testing.assert_array_equal(
            zarr.open_array(OzxStore(path), mode="r", path="0")[:], expected
        )


class TestGuards:
    def test_suffix_warning(self, tmp_path: Path) -> None:
        with pytest.warns(NgioUserWarning, match="should end with"):
            OzxStore(tmp_path / "img.zip", mode="w")

    def test_bad_mode(self, tmp_path: Path) -> None:
        with pytest.raises(NgioValueError, match="Unsupported mode"):
            OzxStore(tmp_path / "img.ozx", mode="rw")  # type: ignore[arg-type]

    def test_one_writer_per_archive(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _write_hierarchy(store)
            with pytest.raises(NgioValueError, match="already open for writing"):
                zarr.open_group(OzxStore(path, mode="r+"), mode="r+")
        with OzxStore(path, mode="r+") as again:
            zarr.open_group(again, mode="r+").attrs["x"] = 1
        _assert_conforming(path)

    def test_io_after_close_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        store = OzxStore(path, mode="w")
        root = zarr.open_group(store, mode="w", zarr_format=3)
        store.close()
        store.close()  # idempotent
        with pytest.raises(NgioValueError, match="closed"):
            root.attrs["x"] = 1
        with pytest.raises(NgioValueError, match="closed"):
            sync(store.get("zarr.json", default_buffer_prototype()))
        with pytest.raises(NgioValueError, match="closed"):
            sync(store.exists("zarr.json"))

    def test_writable_pickle_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            _write_hierarchy(store)
            with pytest.raises(NgioValueError, match="cannot be pickled"):
                pickle.dumps(store)
            with pytest.raises(NgioValueError, match="cannot be pickled"):
                pickle.dumps(store.with_read_only(True))
        copy = pickle.loads(pickle.dumps(OzxStore(path, mode="r")))
        assert sync(copy.exists("zarr.json"))

    def test_refuses_non_conformant_archive(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.ozx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("zarr.json", "{}")
        with pytest.raises(NgioValueError, match="compressed"):
            zarr.open_group(OzxStore(path, mode="r"), mode="r")
        report = check_ozx(path)
        assert not report.conforming
        assert any("STORE" in m for m in report.must)

    def test_gc_finalizes_unclosed_store(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        store = OzxStore(path, mode="w")
        root = zarr.open_group(store, mode="w", zarr_format=3)
        root.attrs.put(ROOT_ATTRS)
        root.create_array("0", shape=(4,), chunks=(4,), dtype="u1")[:] = 5
        del root, store
        gc.collect()
        _assert_conforming(path)
        assert int(zarr.open_array(OzxStore(path), mode="r", path="0")[0]) == 5

    def test_exit_finalizes_unclosed_store(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        script = (
            "import zarr\nfrom ngio.utils._ozx import OzxStore\n"
            f"store = OzxStore({str(path)!r}, mode='w')\n"
            "root = zarr.open_group(store, mode='w', zarr_format=3)\n"
            "root.attrs.put({'ome': {'version': '0.5'}})\n"
            "root.create_array('0', shape=(4,), chunks=(4,), dtype='u1')[:] = 5\n"
        )
        proc = subprocess.run(
            [sys.executable, "-W", "error", "-c", script],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        _assert_conforming(path)


class TestStore:
    def test_read_only_view_shares_live_session(self, tmp_path: Path) -> None:
        path = tmp_path / "img.ozx"
        with OzxStore(path, mode="w") as store:
            root = zarr.open_group(store, mode="w", zarr_format=3)
            root.attrs.put({"ome": {"version": "0.5"}, "live": True})
            view = store.with_read_only(True)
            assert view.read_only and not view.closed
            assert store.with_read_only(False) is store
            # zarr does exactly this when a writable group is reopened read-only
            assert zarr.open_group(view, mode="r").attrs["live"] is True
            root.attrs["live"] = "newer"
            assert zarr.open_group(view, mode="r").attrs["live"] == "newer"
            view.close()  # views never close the archive
            assert not store.closed
            with pytest.raises(NgioValueError, match=r"read-only \.ozx store writable"):
                view.with_read_only(False)
        assert view.closed

    def test_flush_mid_session(self, tmp_path: Path) -> None:
        # `flush` uses private zipfile machinery; this pins its behaviour.
        path = tmp_path / "f.ozx"
        meta = b'{"attributes":{"ome":{"version":"0.5"}}}'
        with _OzxZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0/c/0", b"x" * 100)
            zf.writestr("zarr.json", meta)
            zf.flush()
            assert _names(path) == ["zarr.json", "0/c/0"]
            zf.tombstone(
                "0/c/0"
            )  # the directory shrinks: no stale end record may remain
            zf.tombstone("zarr.json")
            zf.writestr("zarr.json", meta)
            zf.flush()
            assert _names(path) == ["zarr.json"]
            zf.writestr("1/c/0", b"y")
        with zipfile.ZipFile(path) as rd:
            assert rd.comment == rfc9_comment("0.5")
            assert rd.namelist() == ["zarr.json", "1/c/0"]
            assert rd.read("1/c/0") == b"y"
        assert check_ozx(path).conforming

    def test_dedupes_archive_from_stock_zipstore(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.ozx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("zarr.json", b'{"v":1}')
            with pytest.warns(UserWarning, match="Duplicate name"):
                zf.writestr("zarr.json", b'{"v":2}')
        assert check_ozx(path).duplicates == ["zarr.json"]
        with OzxStore(path, mode="r+") as store:
            buf = sync(store.get("zarr.json", default_buffer_prototype()))
            assert buf is not None and buf.to_bytes() == b'{"v":2}'
        assert check_ozx(path).duplicates == []
        assert _names(path) == ["zarr.json"]

    def test_list_dir_and_set_if_not_exists(self, tmp_path: Path) -> None:
        with OzxStore(tmp_path / "img.ozx", mode="w") as store:
            _write_hierarchy(store)
            assert sorted(sync(_collect(store.list_dir("")))) == [
                "0",
                "labels",
                "zarr.json",
            ]
            assert sorted(sync(_collect(store.list_dir("labels")))) == [
                "a",
                "zarr.json",
            ]
            proto = default_buffer_prototype()
            sync(store.set_if_not_exists("zarr.json", proto.buffer.from_bytes(b"nope")))
            buf = sync(store.get("zarr.json", proto))
            assert buf is not None and buf.to_bytes() != b"nope"


async def _collect(it):
    return [k async for k in it]
