"""Ngio's zarr store wrapper: retry behavior + uniform store services.

`NgioStore` is the single place in ngio that dispatches on concrete zarr
store types (local, fsspec, memory, zip). Everything else should ask the
store for the service it needs (`full_url`, `get_mapper`, ...) instead of
isinstance-checking stores.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import warnings
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    MutableMapping,
)
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import fsspec
from zarr.abc.store import ByteRequest, Store
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, MemoryStore, WrapperStore, ZipStore
from zarr.storage._common import make_store

from ngio.config import RetryConfig, get_config
from ngio.utils._errors import NgioValueError
from ngio.utils._retry import aretry_call
from ngio.utils._warnings import NgioUserWarning

if TYPE_CHECKING:
    from zarr.abc.buffer import Buffer
    from zarr.core.buffer import BufferPrototype

logger = logging.getLogger(f"ngio:{__name__}")

T = TypeVar("T")

StoreType = Literal["local", "fsspec", "memory", "zip", "other"]

_KNOWN_STORES = (LocalStore, FsspecStore, MemoryStore, ZipStore)


def _make_sync_fs(fs: fsspec.AbstractFileSystem) -> fsspec.AbstractFileSystem:
    fs_dict = json.loads(fs.to_json())
    fs_dict["asynchronous"] = False
    return fsspec.AbstractFileSystem.from_json(json.dumps(fs_dict))


class NgioStore(WrapperStore[Store]):
    """A zarr store wrapper adding retry behavior and uniform store services.

    All IO methods retry errors according to the `RetryConfig` policy
    (snapshotted from `get_config().io_retry` at construction unless given
    explicitly). With the default policy (`max_retries=0`) every IO call is
    delegated as-is. The instance is picklable, so the policy travels with
    the store into dask task graphs.
    """

    _store: Store
    _retry: RetryConfig

    def __init__(self, store: Store, retry: RetryConfig | None = None) -> None:
        if isinstance(store, NgioStore):
            raise NgioValueError(
                "The store is already an NgioStore. Use NgioStore.ensure() "
                "to wrap stores idempotently."
            )
        if not isinstance(store, _KNOWN_STORES):
            warnings.warn(
                f"Store type {type(store)} is not explicitly supported. "
                f"Supported types are: {_KNOWN_STORES}. "
                "Proceeding, but this may lead to unexpected behavior.",
                NgioUserWarning,
                stacklevel=2,
            )
        super().__init__(store)
        if retry is None:
            retry = get_config().io_retry.model_copy(deep=True)
        self._retry = retry

    # ------------------------------------------------------------------
    # Construction / normalization
    # ------------------------------------------------------------------

    @classmethod
    def from_any(
        cls,
        store: str | Path | fsspec.mapping.FSMap | dict | Store | NgioStore,
        mode: Literal["r", "r+", "w", "w-", "a"] | None = None,
        retry: RetryConfig | None = None,
    ) -> NgioStore:
        """Normalize any supported store-like object into an `NgioStore`.

        Delegates the normalization to zarr (`str`/`Path` become a
        `LocalStore` or, for URIs, an `FsspecStore`; `dict` becomes a
        `MemoryStore` sharing the dict; `FSMap` becomes an `FsspecStore`).
        An existing `NgioStore` is returned unchanged.
        """
        if isinstance(store, NgioStore):
            return store
        if not isinstance(store, Store):
            store = sync(make_store(store, mode=mode))
        return cls(store, retry=retry)

    @classmethod
    def ensure(
        cls, store: Store | NgioStore, retry: RetryConfig | None = None
    ) -> NgioStore:
        """Wrap a zarr store in an `NgioStore`, idempotently."""
        if isinstance(store, NgioStore):
            return store
        return cls(store, retry=retry)

    def _with_store(self, store: Store) -> NgioStore:
        return type(self)(store=store, retry=self._retry)

    def __eq__(self, value: object) -> bool:
        return type(value) is type(self) and self._store == value._store  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return str(self._store)

    def __repr__(self) -> str:
        return f"NgioStore({self._store!r})"

    # ------------------------------------------------------------------
    # Store services
    # ------------------------------------------------------------------

    @property
    def retry_policy(self) -> RetryConfig:
        """Return the retry policy snapshotted at construction."""
        return self._retry

    @property
    def store_type(self) -> StoreType:
        """Return a discriminator for the wrapped concrete store type."""
        if isinstance(self._store, LocalStore):
            return "local"
        if isinstance(self._store, FsspecStore):
            return "fsspec"
        if isinstance(self._store, MemoryStore):
            return "memory"
        if isinstance(self._store, ZipStore):
            return "zip"
        return "other"

    @property
    def is_local(self) -> bool:
        return self.store_type == "local"

    @property
    def local_root(self) -> Path | None:
        """Return the root directory for local stores, `None` otherwise."""
        if isinstance(self._store, LocalStore):
            return Path(self._store.root)
        return None

    @property
    def memory_dict(self) -> MutableMapping[str, Buffer]:
        """Return the backing dict of an in-memory store.

        Raises:
            NgioValueError: If the store is not in-memory.
        """
        if isinstance(self._store, MemoryStore):
            return self._store._store_dict
        raise NgioValueError(
            f"The store is not in-memory (store_type={self.store_type})."
        )

    def full_url(self, path: str = "") -> str | None:
        """Return the full URL/path of `path` inside the store.

        Returns `None` for stores without a meaningful URL (e.g. in-memory).
        """
        store = self._store
        if isinstance(store, LocalStore):
            return (store.root / path).as_posix()
        if isinstance(store, FsspecStore):
            return f"{store.path}/{path}"
        if isinstance(store, ZipStore):
            return (store.path / path).as_posix()
        if isinstance(store, MemoryStore):
            return None
        logger.warning(f"Cannot determine full URL for store type {type(store)}.")
        return None

    def sync_fs_and_path(
        self, path: str = ""
    ) -> tuple[fsspec.AbstractFileSystem, str]:
        """Return a synchronous fsspec filesystem and full path for `path`.

        Raises:
            NgioValueError: If the store is not backed by a filesystem.
        """
        store = self._store
        if isinstance(store, LocalStore):
            return fsspec.filesystem("file"), (store.root / path).as_posix()
        if isinstance(store, FsspecStore):
            return _make_sync_fs(store.fs), f"{store.path}/{path}"
        raise NgioValueError(
            "Cannot build a filesystem for store type "
            f"{self.store_type}; only local and fsspec stores are supported."
        )

    def get_mapper(self, path: str = "") -> fsspec.mapping.FSMap:
        """Return a synchronous fsspec mapper for `path` inside the store."""
        fs, full_path = self.sync_fs_and_path(path)
        return fs.get_mapper(full_path)

    # `builtins.list`: a plain `list[...]` annotation would resolve to the
    # `list` store method defined below.
    def list_dir_collected(self, prefix: str) -> builtins.list[str]:
        """Return the keys directly under `prefix` as a list, synchronously."""

        async def _collect() -> list[str]:
            return [key async for key in self.list_dir(prefix)]

        return sync(_collect())

    # ------------------------------------------------------------------
    # Retried IO
    # ------------------------------------------------------------------

    async def _io(self, op_name: str, fn: Callable[[], Awaitable[T]]) -> T:
        if self._retry.max_retries == 0:
            return await fn()
        return await aretry_call(fn, self._retry, op_name=op_name)

    def _retried_list(
        self, op_name: str, factory: Callable[[], AsyncIterator[str]]
    ) -> AsyncIterator[str]:
        # A partially-consumed generator cannot be safely retried, so the
        # listing is materialized inside the retried call and re-yielded.
        if self._retry.max_retries == 0:
            return factory()

        async def gen() -> AsyncGenerator[str, None]:
            async def collect() -> list[str]:
                return [key async for key in factory()]

            for key in await aretry_call(collect, self._retry, op_name=op_name):
                yield key

        return gen()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        return await self._io(
            f"get '{key}'", lambda: self._store.get(key, prototype, byte_range)
        )

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> builtins.list[Buffer | None]:
        # Retrying re-issues the whole batch; gets are pure so this is safe.
        key_ranges = list(key_ranges)
        return await self._io(
            "get_partial_values",
            lambda: self._store.get_partial_values(prototype, key_ranges),
        )

    async def exists(self, key: str) -> bool:
        return await self._io(f"exists '{key}'", lambda: self._store.exists(key))

    async def set(self, key: str, value: Buffer) -> None:
        return await self._io(f"set '{key}'", lambda: self._store.set(key, value))

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        return await self._io(
            f"set_if_not_exists '{key}'",
            lambda: self._store.set_if_not_exists(key, value),
        )

    async def _get_many(
        self,
        requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]],
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        # Route through self.get so each item is retried individually; the
        # inner store's default would call its own un-retried get.
        for key, prototype, byte_range in requests:
            yield key, await self.get(key, prototype, byte_range)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        # Route through self.set so each item is retried individually.
        await asyncio.gather(*starmap(self.set, values))

    async def delete(self, key: str) -> None:
        return await self._io(f"delete '{key}'", lambda: self._store.delete(key))

    async def delete_dir(self, prefix: str) -> None:
        return await self._io(
            f"delete_dir '{prefix}'", lambda: self._store.delete_dir(prefix)
        )

    async def is_empty(self, prefix: str) -> bool:
        return await self._io(
            f"is_empty '{prefix}'", lambda: self._store.is_empty(prefix)
        )

    async def clear(self) -> None:
        return await self._io("clear", self._store.clear)

    async def getsize(self, key: str) -> int:
        # Must delegate: the Store ABC default downloads the whole object.
        return await self._io(f"getsize '{key}'", lambda: self._store.getsize(key))

    async def getsize_prefix(self, prefix: str) -> int:
        return await self._io(
            f"getsize_prefix '{prefix}'", lambda: self._store.getsize_prefix(prefix)
        )

    @property
    def supports_consolidated_metadata(self) -> bool:
        return self._store.supports_consolidated_metadata

    def list(self) -> AsyncIterator[str]:
        return self._retried_list("list", self._store.list)

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._retried_list(
            f"list_prefix '{prefix}'", lambda: self._store.list_prefix(prefix)
        )

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._retried_list(
            f"list_dir '{prefix}'", lambda: self._store.list_dir(prefix)
        )
