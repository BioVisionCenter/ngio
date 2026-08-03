from __future__ import annotations

import errno
import pickle
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import fsspec
import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, MemoryStore, WrapperStore, ZipStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from zarr.abc.buffer import Buffer
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype

import ngio.utils._retry as retry_mod
from ngio.config import ConstantBackoff, RetryConfig
from ngio.utils import (
    NgioFileExistsError,
    NgioStore,
    NgioUserWarning,
    NgioValueError,
)

_RETRY = RetryConfig(
    max_retries=3,
    retry_on=["OSError"],
    backoff=ConstantBackoff(delay_s=0.0, jitter=False),
)


class FlakyMemoryStore(MemoryStore):
    """A MemoryStore that raises OSError n times per method before working."""

    def __init__(self, fail_times: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.attempts: Counter[str] = Counter()

    def _flake(self, method: str) -> None:
        self.attempts[method] += 1
        if self.attempts[method] <= self.fail_times:
            raise OSError(f"flaky {method}")

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self._flake("get")
        return await super().get(key, prototype, byte_range)

    async def set(
        self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None
    ) -> None:
        self._flake("set")
        return await super().set(key, value, byte_range)

    async def exists(self, key: str) -> bool:
        self._flake("exists")
        return await super().exists(key)

    async def delete(self, key: str) -> None:
        self._flake("delete")
        return await super().delete(key)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        self._flake("list_dir")
        return super().list_dir(prefix)


class TestFromAny:
    def test_str_path(self, tmp_path):
        store = NgioStore.from_any(str(tmp_path))
        assert isinstance(store._store, LocalStore)

    def test_path(self, tmp_path):
        store = NgioStore.from_any(tmp_path)
        assert isinstance(store._store, LocalStore)

    def test_dict_shares_state(self):
        backing: dict = {}
        store_a = NgioStore.from_any(backing, mode="a")
        assert isinstance(store_a._store, MemoryStore)
        group = zarr.open_group(store=store_a, mode="a")
        group.attrs["marker"] = 1
        store_b = NgioStore.from_any(backing, mode="a")
        reopened = zarr.open_group(store=store_b, mode="r")
        assert reopened.attrs["marker"] == 1

    def test_fsmap(self, tmp_path):
        mapper = fsspec.get_mapper(str(tmp_path))
        store = NgioStore.from_any(mapper)
        assert isinstance(store._store, FsspecStore)

    def test_store_instance(self):
        store = NgioStore.from_any(MemoryStore())
        assert isinstance(store._store, MemoryStore)

    def test_idempotent_on_ngio_store(self):
        store = NgioStore.from_any(MemoryStore())
        assert NgioStore.from_any(store) is store
        assert NgioStore.ensure(store) is store

    def test_double_wrap_raises(self):
        store = NgioStore(MemoryStore())
        with pytest.raises(NgioValueError, match="already an NgioStore"):
            NgioStore(store)

    def test_exotic_store_warns_via_from_any(self):
        exotic = WrapperStore(MemoryStore())
        with pytest.warns(NgioUserWarning, match="not explicitly supported"):
            store = NgioStore.from_any(exotic)
        assert store.store_type == "other"

    def test_exotic_store_no_warning_via_ensure(self, recwarn):
        store = NgioStore.ensure(WrapperStore(MemoryStore()))
        assert store.store_type == "other"
        assert len(recwarn) == 0


class TestServices:
    def test_local(self, tmp_path):
        store = NgioStore.from_any(tmp_path)
        assert store.store_type == "local"
        assert store.is_local
        assert store.local_root == tmp_path
        assert store.full_url("sub/group") == (tmp_path / "sub/group").as_posix()
        with pytest.raises(NgioValueError, match="not in-memory"):
            _ = store.memory_dict

    def test_memory(self):
        backing: dict = {}
        store = NgioStore.from_any(backing)
        assert store.store_type == "memory"
        assert not store.is_local
        assert store.local_root is None
        assert store.full_url("sub") is None
        assert store.memory_dict is backing

    def test_zip(self, tmp_path):
        store = NgioStore(ZipStore(tmp_path / "data.zip", mode="w"))
        zarr.open_group(store=store, mode="a")
        assert store.store_type == "zip"
        expected = (tmp_path / "data.zip" / "sub").as_posix()
        assert store.full_url("sub") == expected
        with pytest.raises(NgioValueError, match="Cannot build a filesystem"):
            store.sync_fs_and_path("sub")
        store._store.close()

    def test_fsspec(self, tmp_path):
        store = NgioStore.from_any(fsspec.get_mapper(str(tmp_path)))
        assert store.store_type == "fsspec"
        fs, full_path = store.sync_fs_and_path("sub")
        assert full_path.endswith("/sub")
        assert not getattr(fs, "asynchronous", False)

    def test_get_mapper_roundtrip(self, tmp_path):
        store = NgioStore.from_any(tmp_path)
        mapper = store.get_mapper("sub")
        mapper["key"] = b"value"
        assert (tmp_path / "sub" / "key").read_bytes() == b"value"

    def test_list_dir_collected(self):
        store = NgioStore(MemoryStore())
        group = zarr.open_group(store=store, mode="a")
        group.create_group("child")
        keys = store.list_dir_collected("")
        assert "zarr.json" in keys
        assert "child" in keys

    def test_full_url_parity_with_handler(self, tmp_path):
        # Mirrors ZarrGroupHandler.full_url expectations ahead of phase 3.
        from ngio.utils import ZarrGroupHandler

        handler = ZarrGroupHandler(store=tmp_path / "data.zarr", mode="a")
        store = NgioStore.from_any(tmp_path / "data.zarr")
        assert store.full_url(handler.group.path) == handler.full_url


class TestRetriedIO:
    def test_flaky_store_recovers(self):
        flaky = FlakyMemoryStore(fail_times=1)
        store = NgioStore(flaky, retry=_RETRY)
        group = zarr.open_group(store=store, mode="a")
        group.attrs["marker"] = 42
        reopened = zarr.open_group(store=store, mode="r")
        assert reopened.attrs["marker"] == 42
        assert flaky.attempts["get"] > flaky.fail_times
        assert flaky.attempts["set"] > flaky.fail_times

    def test_default_policy_propagates_error(self):
        flaky = FlakyMemoryStore(fail_times=1)
        store = NgioStore(flaky, retry=RetryConfig())
        with pytest.raises(OSError, match="flaky"):
            sync(store.exists("key"))
        assert flaky.attempts["exists"] == 1

    def test_non_matching_marker_propagates_error(self):
        flaky = FlakyMemoryStore(fail_times=1)
        retry = _RETRY.model_copy(update={"retry_on": ["TimeoutError"]})
        store = NgioStore(flaky, retry=retry)
        with pytest.raises(OSError, match="flaky"):
            sync(store.exists("key"))
        assert flaky.attempts["exists"] == 1

    def test_retries_exhausted(self):
        flaky = FlakyMemoryStore(fail_times=10)
        store = NgioStore(flaky, retry=_RETRY)
        with pytest.raises(OSError, match="flaky"):
            sync(store.exists("key"))
        assert flaky.attempts["exists"] == _RETRY.max_retries + 1

    def test_direct_store_methods_retry(self):
        flaky = FlakyMemoryStore(fail_times=1)
        store = NgioStore(flaky, retry=_RETRY)
        assert sync(store.exists("nope")) is False
        assert flaky.attempts["exists"] == 2

    def test_list_dir_retries(self):
        flaky = FlakyMemoryStore(fail_times=1)
        store = NgioStore(flaky, retry=_RETRY)

        async def collect():
            return [k async for k in store.list_dir("")]

        assert sync(collect()) == []
        assert flaky.attempts["list_dir"] == 2

    def test_get_many_routes_through_retried_get(self):
        from zarr.core.buffer import default_buffer_prototype

        flaky = FlakyMemoryStore(fail_times=1)
        store = NgioStore(flaky, retry=_RETRY)
        prototype = default_buffer_prototype()

        async def collect():
            requests = [("a", prototype, None), ("b", prototype, None)]
            return [item async for item in store._get_many(requests)]

        results = dict(sync(collect()))
        assert results == {"a": None, "b": None}
        assert flaky.attempts["get"] == 3  # 1 failure + 2 successes


class SharingViolationStore(MemoryStore):
    """A MemoryStore raising a Windows sharing violation n times per method."""

    def __init__(self, fail_times: int = 1, winerror: int = 5, exc=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.winerror = winerror
        self.exc = exc
        self.attempts: Counter[str] = Counter()

    def _flake(self, method: str) -> None:
        self.attempts[method] += 1
        if self.attempts[method] > self.fail_times:
            return
        if self.exc is not None:
            raise self.exc
        exc = PermissionError(13, "Access is denied")
        exc.winerror = self.winerror
        raise exc

    async def set(
        self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None
    ) -> None:
        self._flake("set")
        return await super().set(key, value, byte_range)

    async def exists(self, key: str) -> bool:
        self._flake("exists")
        return await super().exists(key)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        self._flake("list_dir")
        return super().list_dir(prefix)


class TestSharingViolationRetry:
    """Windows sharing violations are absorbed independently of `io_retry`."""

    @pytest.fixture(autouse=True)
    def _simulate_windows(self, monkeypatch):
        monkeypatch.setattr(retry_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(
            retry_mod,
            "_SHARING_VIOLATION_BACKOFF",
            ConstantBackoff(delay_s=0.0, jitter=False),
        )

    def test_recovers_from_the_reader_side_shape(self, tmp_path):
        # What `open()` raises on Windows for a delete-pending target: errno
        # only, no winerror. See `_retry.is_sharing_violation`.
        target = tmp_path / "zarr.json"
        target.write_text("{}")
        flaky = SharingViolationStore(
            fail_times=1,
            exc=PermissionError(errno.EACCES, "Permission denied", str(target)),
        )
        store = NgioStore(flaky, retry=RetryConfig())
        assert sync(store.exists("nope")) is False
        assert flaky.attempts["exists"] == 2

    @pytest.mark.parametrize("winerror", [5, 32, 33])
    def test_recovers_with_retries_disabled(self, winerror):
        flaky = SharingViolationStore(fail_times=1, winerror=winerror)
        store = NgioStore(flaky, retry=RetryConfig())
        assert sync(store.exists("nope")) is False
        assert flaky.attempts["exists"] == 2

    def test_exhaustion_raises_the_original_error(self):
        flaky = SharingViolationStore(fail_times=100)
        store = NgioStore(flaky, retry=RetryConfig())
        with pytest.raises(PermissionError) as excinfo:
            sync(store.exists("nope"))
        assert excinfo.value.winerror == 5
        assert flaky.attempts["exists"] == retry_mod._SHARING_VIOLATION_ATTEMPTS

    def test_not_retried_off_windows(self, monkeypatch):
        monkeypatch.setattr(retry_mod, "_IS_WINDOWS", False)
        flaky = SharingViolationStore(fail_times=1)
        store = NgioStore(flaky, retry=RetryConfig())
        with pytest.raises(PermissionError):
            sync(store.exists("nope"))
        assert flaky.attempts["exists"] == 1

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError("Access Denied"),  # the s3fs 403 shape
            NgioFileExistsError("already there"),
            OSError("plain"),
        ],
    )
    def test_other_errors_propagate_immediately(self, exc):
        flaky = SharingViolationStore(fail_times=1, exc=exc)
        store = NgioStore(flaky, retry=RetryConfig())
        with pytest.raises(type(exc)):
            sync(store.exists("nope"))
        assert flaky.attempts["exists"] == 1

    def test_writes_recover(self):
        flaky = SharingViolationStore(fail_times=1)
        store = NgioStore(flaky, retry=RetryConfig())
        group = zarr.open_group(store=store, mode="a")
        group.attrs["marker"] = 42
        assert flaky.attempts["set"] > flaky.fail_times

    def test_listing_recovers(self):
        flaky = SharingViolationStore(fail_times=1)
        store = NgioStore(flaky, retry=RetryConfig())

        async def collect():
            return [k async for k in store.list_dir("")]

        assert sync(collect()) == []
        assert flaky.attempts["list_dir"] == 2

    def test_composes_multiplicatively_with_io_retry(self):
        flaky = SharingViolationStore(fail_times=1000)
        retry = RetryConfig(
            max_retries=3,
            retry_on=["PermissionError"],
            backoff=ConstantBackoff(delay_s=0.0, jitter=False),
        )
        store = NgioStore(flaky, retry=retry)
        with pytest.raises(PermissionError):
            sync(store.exists("nope"))
        assert flaky.attempts["exists"] == retry_mod._SHARING_VIOLATION_ATTEMPTS * (
            retry.max_retries + 1
        )


class TestStoreBehavior:
    def test_pickle_roundtrip(self, tmp_path):
        store = NgioStore.from_any(tmp_path / "data.zarr", mode="a", retry=_RETRY)
        group = zarr.open_group(store=store, mode="a")
        group.attrs["marker"] = 7
        restored = pickle.loads(pickle.dumps(store))
        assert isinstance(restored, NgioStore)
        assert restored.retry_policy == _RETRY
        reopened = zarr.open_group(store=restored, mode="r")
        assert reopened.attrs["marker"] == 7

    def test_with_read_only_preserves_policy(self):
        store = NgioStore(MemoryStore(), retry=_RETRY)
        read_only = store.with_read_only(True)
        assert isinstance(read_only, NgioStore)
        assert read_only.read_only is True
        assert read_only.retry_policy is _RETRY

    def test_eq(self):
        inner = MemoryStore()
        assert NgioStore(inner) == NgioStore(inner)
        assert NgioStore(inner) != inner
        buf = default_buffer_prototype().buffer.from_bytes(b"x")
        other = MemoryStore(store_dict={"key": buf})
        assert NgioStore(other) != NgioStore(inner)

    def test_dask_read_through_store(self, tmp_path):
        store = NgioStore.from_any(tmp_path / "arr.zarr", mode="a")
        group = zarr.open_group(store=store, mode="a")
        arr = group.create_array("x", shape=(8, 8), chunks=(4, 4), dtype="uint16")
        data = np.arange(64, dtype="uint16").reshape(8, 8)
        arr[:] = data
        result = da.from_zarr(arr).compute()
        np.testing.assert_array_equal(result, data)

    def test_getsize_delegates(self):
        store = NgioStore(MemoryStore())
        group = zarr.open_group(store=store, mode="a")
        group.attrs["marker"] = 1
        assert sync(store.getsize("zarr.json")) > 0


class TestRetryEndToEnd:
    """Retry through the full ngio stack, driven by the global config."""

    def test_handler_retries_via_global_config(self, monkeypatch):
        from ngio.config import get_config
        from ngio.utils import ZarrGroupHandler

        monkeypatch.setattr(get_config(), "io_retry", _RETRY)
        flaky = FlakyMemoryStore(fail_times=1)
        handler = ZarrGroupHandler(store=flaky, mode="a")
        handler.write_attrs({"marker": 3})
        assert handler.load_attrs() == {"marker": 3}
        assert flaky.attempts["get"] > flaky.fail_times

    def test_handler_fails_without_retry_config(self):
        from ngio.utils import ZarrGroupHandler

        flaky = FlakyMemoryStore(fail_times=1)
        with pytest.raises(OSError, match="flaky"):
            ZarrGroupHandler(store=flaky, mode="a")

    def test_user_group_is_rewrapped(self, tmp_path):
        from ngio.utils import ZarrGroupHandler

        group = zarr.open_group(store=tmp_path / "user.zarr", mode="a")
        assert not isinstance(group.store, NgioStore)
        handler = ZarrGroupHandler(store=group, mode="a")
        assert isinstance(handler.group.store, NgioStore)
        # and it is not double wrapped when passed around again
        handler2 = ZarrGroupHandler(store=handler.group, mode="a")
        assert isinstance(handler2.group.store, NgioStore)
        assert not isinstance(handler2.group.store._store, NgioStore)


class TestMakeStoreCompat:
    """Canary for zarr's private make_store: fails loudly on a zarr bump."""

    @pytest.mark.parametrize(
        "store_like,expected",
        [
            ("{tmp_path}", LocalStore),
            (Path("{tmp_path}"), LocalStore),
            ({}, MemoryStore),
            ("memory://test", FsspecStore),
        ],
    )
    def test_input_type_mapping(self, tmp_path, store_like, expected):
        if isinstance(store_like, str) and "{tmp_path}" in store_like:
            store_like = store_like.format(tmp_path=tmp_path)
        elif isinstance(store_like, Path):
            store_like = tmp_path
        store = NgioStore.from_any(store_like)
        assert isinstance(store._store, expected)

    def test_fsmap_maps_to_fsspec(self, tmp_path):
        mapper = fsspec.get_mapper(str(tmp_path))
        assert isinstance(NgioStore.from_any(mapper)._store, FsspecStore)
