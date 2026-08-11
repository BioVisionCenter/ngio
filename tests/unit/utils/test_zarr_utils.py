import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import fsspec.implementations.http
import numpy as np
import pytest
import zarr
from filelock import BaseFileLock
from zarr.storage import LocalStore

from ngio.utils import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioUserWarning,
    NgioValueError,
    ZarrGroupHandler,
    _retry,
    open_group_wrapper,
)


@pytest.mark.parametrize("cache", [True, False])
def test_group_handler_creation(tmp_path: Path, cache: bool):
    from ngio.utils import NgioStore

    store = tmp_path / "test_group_handler_creation.zarr"
    handler = ZarrGroupHandler(store=store, cache=cache, mode="a")

    _store = handler.group.store
    assert isinstance(_store, NgioStore)
    assert isinstance(_store._store, LocalStore)
    assert _store.local_root == store
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
    from ngio.utils import NgioStore

    store = tmp_path / "test_group_handler_from_group.zarr"
    group = zarr.group(store=store, overwrite=True)

    handler = ZarrGroupHandler(store=group, cache=True, mode="a")
    # The group is reopened on an NgioStore wrapping the original store,
    # so compare path and underlying store rather than group identity.
    assert isinstance(handler.group.store, NgioStore)
    assert handler.group.store._store == group.store
    assert handler.group.path == group.path


def test_cache_true_holds_until_refreshed(tmp_path: Path):
    """`cache=True` means "hold it for my lifetime, I am the only writer".

    A write that goes around the handler is therefore *not* visible until
    `refresh()`. This is the whole content of the flag, and it used to be
    invisible: `load_attrs` reopened the group unconditionally, so caching was
    inert for metadata and an outside write showed up regardless.
    """
    store = tmp_path / "cache_semantics.zarr"
    group = zarr.group(store=store, overwrite=True)

    cached = ZarrGroupHandler(store=group, cache=True, mode="a")
    uncached = ZarrGroupHandler(store=group, cache=False, mode="a")
    assert cached.load_attrs() == {}

    group.attrs["marker"] = 1

    assert uncached.load_attrs() == {"marker": 1}
    assert cached.load_attrs() == {}

    cached.refresh()
    assert cached.load_attrs() == {"marker": 1}


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


def _append_under_lock(args: tuple[str, int, bool]) -> int:
    """Append one item to a shared attrs list, in a worker process.

    Module level and picklable by reference, which is what `spawn` needs.
    Goes through `locked()` rather than `lock` because that is what refreshes
    cached metadata around the critical section — with `cache=True` and a bare
    `lock`, the read below could be served from before the lock was taken and
    the write would drop the previous holder's item.
    """
    zarr_store, i, cache = args
    handler = ZarrGroupHandler(zarr_store, cache=cache, mode="a")
    # Prime the cache *before* the lock. Without this the handler is freshly
    # built and its cache empty, so the read below would go to the store
    # whatever the policy, and the cached half of this test would prove nothing.
    handler.load_attrs()
    with handler.locked():
        attrs = handler.load_attrs()
        attrs["test_list"].append(i)
        handler.write_attrs(attrs, overwrite=False)
    return os.getpid()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the lock is only best-effort on Windows, so no-lost-update cannot hold",
)
@pytest.mark.parametrize("cache", [False, True])
def test_multiprocessing_safety(tmp_path: Path, cache: bool):
    """The lock holds between real processes, which is what it claims.

    Runs in separate processes rather than dask's threaded scheduler: this test
    is the one that has to notice a lock which only excludes threads, and until
    now it ran entirely in one interpreter despite its name. `spawn`, not
    `fork` — zarr keeps an IO event loop thread and forking one warns.

    `cache=True` is the interesting half. Caching used to be *refused* whenever
    a lock was asked for, on the grounds that a cached read-modify-write would
    silently drop a concurrent update. That is a real hazard, and it is what
    `locked()` invalidating on entry and exit exists to remove — so this asserts
    the replacement is at least as strong as the refusal it replaced.
    """
    zarr_store = tmp_path / "test_multiprocessing_safety.zarr"

    handler = ZarrGroupHandler(zarr_store, cache=cache, mode="w")
    handler.write_attrs({"test_list": []}, overwrite=True)

    num_items = 40
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pids = set(
            pool.map(
                _append_under_lock,
                [(str(zarr_store), i, cache) for i in range(num_items)],
            )
        )

    # Guard against this silently becoming a single-process test again.
    assert os.getpid() not in pids
    assert len(pids) > 1

    handler.refresh()
    _, counts = np.unique(handler.load_attrs()["test_list"], return_counts=True)
    assert len(counts) == num_items
    assert np.all(counts == 1)

    assert handler.lock_path is not None


def test_a_cached_handler_can_take_the_lock(tmp_path: Path):
    """Caching and locking used to be mutually exclusive. They compose now."""
    handler = ZarrGroupHandler(tmp_path / "cached.zarr", cache=True, mode="a")

    lock_path, lock = handler._create_lock()
    assert isinstance(lock, BaseFileLock)
    with handler.locked():
        assert lock_path.exists()


def test_lock_warns_on_windows(tmp_path: Path, monkeypatch):
    """On Windows the lock warns and is still handed out, not refused.

    Uncontended it is exclusive there too, so refusing would break every
    single-writer caller to protect against a race they are not running.
    Simulated, so the warning is exercised on every platform.
    """
    handler = ZarrGroupHandler(tmp_path / "win.zarr", cache=False, mode="a")

    monkeypatch.setattr(_retry, "_IS_WINDOWS", True)
    with pytest.warns(NgioUserWarning, match="not exclusive on Windows"):
        lock_path, lock = handler._create_lock()

    assert isinstance(lock, BaseFileLock)
    with lock:
        assert lock_path.exists()


def test_windows_warning_does_not_mask_the_validations(monkeypatch):
    """A store that cannot be locked at all still raises, on Windows too.

    The warning sits after the store-type guard deliberately. Emitted first, it
    would turn into the raised exception under `-W error` — so Windows would
    report `NgioUserWarning` for a remote store where every other platform
    reports `NgioValueError`.
    """
    monkeypatch.setattr(_retry, "_IS_WINDOWS", True)
    handler = ZarrGroupHandler({}, cache=False, mode="a")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NgioValueError, match="needs to be a LocalStore"):
            handler._create_lock()


def test_lock_paths_live_outside_the_store(tmp_path: Path):
    """Lock files never land inside the store, and never collide.

    `with_suffix('.lock')` used to put a well's lock at `store.zarr/C/03.lock`
    — a non-Zarr entry inside the store — and mapped `foo.bar` and `foo.baz`
    onto one `foo.lock`, so two unrelated groups shared a critical section.
    """
    store = tmp_path / "store.zarr"
    handler = ZarrGroupHandler(store, cache=False, mode="a")
    for path in ("C/03", "C/04", "foo.bar", "foo.baz"):
        handler.create_group(path)

    lock_paths = {
        path: handler.get_handler(path).lock_path
        for path in ("C/03", "C/04", "foo.bar", "foo.baz")
    }
    lock_paths["<root>"] = handler.lock_path

    assert len(set(lock_paths.values())) == len(lock_paths)
    for path, lock_path in lock_paths.items():
        assert store not in lock_path.parents, f"{path} locks inside the store"
        # ngio creates the directory itself: no `filelock` makes it at
        # construction, and versions below 3.12.3 do not make it on acquire
        # either — which failed only the min-deps CI leg, never the dev env.
        assert lock_path.parent.is_dir(), f"{path} lock directory not created"

    # Acquiring must not leave anything behind inside the store either.
    with handler.get_handler("C/03").lock:
        pass
    assert list(store.rglob("*.lock")) == []


def test_get_group_adopts_a_concurrently_created_group(tmp_path: Path, monkeypatch):
    """`create_mode=True` is get-or-create, so losing the create race is fine.

    Deterministic stand-in for a real race: the existence probe is forced to
    miss once, exactly as it would if another worker created the group just
    after the probe ran. Before this was handled, the loser raised
    `NgioFileExistsError` — which is how a plate write failed in CI while four
    workers added images to the same well.
    """
    handler = ZarrGroupHandler(tmp_path / "store.zarr", cache=False, mode="a")
    existing = handler.create_group("C/03")

    real_get = zarr.Group.get
    missed = []

    def miss_once(self, path, default=None):
        if path == "C/03" and not missed:
            missed.append(path)
            return None
        return real_get(self, path, default=default)

    monkeypatch.setattr(zarr.Group, "get", miss_once)

    group = handler.get_group("C/03", create_mode=True)
    assert missed, "the probe was never exercised"
    assert isinstance(group, zarr.Group)
    assert group.path == existing.path

    # A group that genuinely is not there must still be created, and a missing
    # one without create_mode must still raise.
    assert handler.get_group("D/04", create_mode=True).path == "D/04"
    with pytest.raises(NgioFileNotFoundError):
        handler.get_group("E/05", create_mode=False)


@pytest.mark.network
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


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_is_group_listable(monkeypatch: pytest.MonkeyPatch, zarr_format: Literal[2, 3]):
    from zarr.storage import MemoryStore

    from ngio.utils._zarr_utils import is_group_listable

    group = zarr.create_group(store=MemoryStore(), zarr_format=zarr_format)
    labels = group.create_group("labels")
    assert is_group_listable(group)
    assert is_group_listable(labels)

    # A genuinely empty group still lists its own metadata document
    empty_group = zarr.create_group(store=MemoryStore(), zarr_format=zarr_format)
    assert is_group_listable(empty_group)

    # zarr >= 3.1.6 swallows listing errors on some stores and yields an
    # empty listing instead; the group's metadata document is then missing
    async def _empty_list_dir(prefix: str):
        return
        yield

    monkeypatch.setattr(group.store, "list_dir", _empty_list_dir)
    assert not is_group_listable(group)

    async def _raising_list_dir(prefix: str):
        raise FileNotFoundError(prefix)
        yield

    monkeypatch.setattr(group.store, "list_dir", _raising_list_dir)
    assert not is_group_listable(group)


def test_fsspec_copy_refuses_unlistable_source(tmp_path: Path):
    from ngio.utils import NgioStore
    from ngio.utils._zarr_utils import _fsspec_copy

    src_path = tmp_path / "empty_src"
    src_path.mkdir()
    dest_path = tmp_path / "dest"

    with pytest.raises(NgioValueError):
        _fsspec_copy(
            NgioStore(LocalStore(src_path)), "", NgioStore(LocalStore(dest_path)), ""
        )

    # Nothing was written to the destination
    assert not dest_path.exists() or not any(dest_path.iterdir())


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


def test_writes_are_visible_through_a_stale_consolidated_metadata(tmp_path: Path):
    """A store carrying `.zmetadata` must not shadow ngio's own writes.

    zarr leaves consolidated metadata untouched when attributes are written, and
    ngio has no way to refresh someone else's. Trusting it made ngio read back a
    snapshot from before its own writes: a group ngio had just created was
    absent from the very next listing.
    """
    store = tmp_path / "consolidated.zarr"
    group = zarr.create_group(store=store, overwrite=True, zarr_format=2)
    group.attrs.update({"members": []})
    zarr.consolidate_metadata(str(store))
    assert (store / ".zmetadata").exists()

    handler = ZarrGroupHandler(store=store, cache=False, mode="r+")
    handler.write_attrs({"members": ["added_after_consolidation"]})
    handler.create_group("child")

    assert handler.load_attrs()["members"] == ["added_after_consolidation"]
    assert "child" in handler.group
