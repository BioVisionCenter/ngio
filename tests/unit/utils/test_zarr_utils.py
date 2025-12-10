import sys
from pathlib import Path

import dask
import dask.delayed
import fsspec.implementations.http
import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from ngio.utils import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioValueError,
    ZarrGroupHandler,
    open_group_wrapper,
)


@pytest.mark.parametrize("cache", [True, False])
def test_group_handler_creation(tmp_path: Path, cache: bool):
    store = tmp_path / "test_group_handler_creation.zarr"
    handler = ZarrGroupHandler(store=store, cache=cache, mode="a")

    _store = handler.group.store
    assert isinstance(_store, LocalStore)
    assert Path(_store.root.as_posix()) == store
    assert handler.use_cache == cache

    attrs = handler.load_attrs()
    assert attrs == {}
    attrs = {"a": 1, "b": 2, "c": 3}
    handler.write_attrs(attrs)
    assert handler.load_attrs() == attrs
    handler.clean_cache()

    handler.write_attrs({"a": 2}, overwrite=False)
    assert handler.load_attrs()["a"] == 2
    assert handler.load_attrs()["b"] == 2

    handler.write_attrs({"a": 3}, overwrite=True)
    assert handler.load_attrs()["a"] == 3
    assert "b" not in handler.load_attrs()

    new_group = handler.create_group("new_group")

    assert isinstance(new_group, zarr.Group)
    assert isinstance(handler.get_group("new_group"), zarr.Group)

    with pytest.raises(NgioFileExistsError):
        handler.create_group("new_group", overwrite=False)

    # Delete the group
    handler.delete_group("new_group")
    with pytest.raises(NgioFileNotFoundError):
        handler.get_group("new_group")


def test_group_handler_from_group(tmp_path: Path):
    store = tmp_path / "test_group_handler_from_group.zarr"
    group = zarr.group(store=store, overwrite=True)

    handler = ZarrGroupHandler(store=group, cache=True, mode="a")
    assert handler.group == group


def test_group_handler_delete(tmp_path: Path):
    store = tmp_path / "test_group_handler_from_group.zarr"
    group = zarr.group(store=store, overwrite=True)
    group.create_group("to_be_deleted")
    handler = ZarrGroupHandler(store=group, cache=True, mode="a")
    assert isinstance(handler.get_group("to_be_deleted"), zarr.Group)
    handler.delete_group("to_be_deleted")
    with pytest.raises(NgioFileNotFoundError):
        handler.get_group("to_be_deleted")
    assert store.exists()
    handler.delete_self()
    assert not store.exists()

    store = tmp_path / "test_group_handler_from_group.zarr"
    group = zarr.group(store=store, overwrite=True)
    group.create_group("to_be_deleted")
    handler = ZarrGroupHandler(store=group, cache=True, mode="r")
    with pytest.raises(NgioValueError):
        handler.delete_group("to_be_deleted")
    with pytest.raises(NgioValueError):
        handler.delete_self()


def test_group_handler_read(tmp_path: Path):
    store = tmp_path / "test_group_handler_read.zarr"

    group = zarr.create_group(store=store, overwrite=True)
    input_attrs = {"a": 1, "b": 2, "c": 3}
    group.attrs.update(input_attrs)

    group.create_group("group1")
    group.create_array("array1", shape=(10, 10), dtype="int32")

    handler = ZarrGroupHandler(store=store, cache=True, mode="r")

    assert handler.load_attrs() == input_attrs
    assert isinstance(handler.get_array("array1"), zarr.Array)
    assert isinstance(handler.get_group("group1"), zarr.Group)
    assert handler.read_only

    with pytest.raises(NgioFileNotFoundError):
        handler.get_array("array2")

    with pytest.raises(NgioFileNotFoundError):
        handler.get_group("group2")

    with pytest.raises(NgioValueError):
        handler.get_array("group1")

    with pytest.raises(NgioValueError):
        handler.get_group("array1")

    with pytest.raises(NgioValueError):
        handler.write_attrs({"a": 1, "b": 2, "c": 3})


def test_open_fail(tmp_path: Path):
    store = tmp_path / "test_open_fail.zarr"
    group = zarr.create_group(store=store, overwrite=True)

    read_only_group = open_group_wrapper(store=group, mode="r")
    assert read_only_group.read_only

    with pytest.raises(NgioFileExistsError):
        open_group_wrapper(store=store, mode="w-")

    with pytest.raises(NgioFileNotFoundError):
        open_group_wrapper(store=store / "non_existent.zarr", mode="r")

    with pytest.raises(NgioValueError):
        open_group_wrapper(store=read_only_group, mode="w")


def test_multiprocessing_safety(tmp_path: Path):
    zarr_store = tmp_path / "test_multiprocessing_safety.zarr"

    @dask.delayed  # type: ignore
    def add_item(i):
        handler = ZarrGroupHandler(zarr_store, cache=False, mode="a")
        assert handler.lock is not None

        with handler.lock:
            attrs = handler.load_attrs()
            attrs["test_list"].append(i)
            handler.write_attrs(attrs, overwrite=False)

        return i

    handler = ZarrGroupHandler(zarr_store, cache=False, mode="w")
    attrs = handler.load_attrs()
    attrs = {"test_list": []}
    handler.write_attrs(attrs, overwrite=True)

    results = []
    num_items = 1000
    for i in range(num_items):
        results.append(add_item(i))

    dask.compute(*results)  # type: ignore

    _, counts = np.unique(handler.load_attrs()["test_list"], return_counts=True)
    assert len(counts) == num_items
    assert np.all(counts == 1)

    assert handler.lock_path is not None

    if sys.platform.startswith("win"):
        # The file lock path is not removed on Windows
        # for some reason path.exists() returns False
        # even though the file should exist (or at least it does on Mac/Linux)
        return None

    assert handler.lock_path.exists()
    lock_path = handler.lock_path
    handler.remove_lock()
    assert not lock_path.exists()
    handler.remove_lock()

    handler = ZarrGroupHandler(zarr_store, cache=False, mode="w")
    assert handler.lock is not None
    with pytest.raises(NgioValueError):
        # Attempt to remove the lock while it is in use
        with handler.lock:
            handler.remove_lock()

    # If cache is used, raise error when creating lock
    # Since caching creates a temporary local copy
    # which cannot be locked properly
    handler = ZarrGroupHandler(zarr_store, cache=True, mode="r")
    with pytest.raises(NgioValueError):
        handler._create_lock()


def test_remote_storage():
    url = (
        "https://raw.githubusercontent.com/"
        "fractal-analytics-platform/fractal-ome-zarr-examples/"
        "refs/heads/main/v04/"
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
    )

    fs = fsspec.implementations.http.HTTPFileSystem(client_kwargs={})
    store = fs.get_mapper(url)
    handler = ZarrGroupHandler(store=store, cache=True, mode="r")
    assert handler.load_attrs()
    assert isinstance(handler.get_array("0"), zarr.Array)
    assert isinstance(handler.get_group("labels"), zarr.Group)
    assert not handler.is_listable


@pytest.mark.parametrize(
    "src_store,dest_store",
    [
        (Path("src.zarr"), Path("dest.zarr")),
        (Path("src.zarr"), {}),
        (Path("dest.zarr"), {}),
    ],
)
def test_copy_group(tmp_path: Path, src_store, dest_store):
    if isinstance(src_store, Path):
        src_store = tmp_path / src_store
    if isinstance(dest_store, Path):
        dest_store = tmp_path / dest_store

    src_group = zarr.create_group(store=src_store, overwrite=True)
    src_group.attrs.update({"a": 1, "b": 2, "c": 3})
    src_group.create_array("array1", shape=(10, 10), dtype="int32")
    sub_group = src_group.create_group("group1")
    sub_group.create_array("sub_array1", shape=(5, 5), dtype="float32")
    handler = ZarrGroupHandler(store=src_group, cache=False, mode="r")

    dest_group = zarr.group(store=dest_store, overwrite=True)
    handler.copy_group(dest_group=dest_group)
    # Reopen dest group to ensure all data is read from store
    dest_group = zarr.open_group(dest_store, mode="r")
    assert dest_group.attrs.asdict() == src_group.attrs.asdict()
    assert "array1" in dest_group
    assert "group1" in dest_group
    assert "sub_array1" in dest_group["group1"]
