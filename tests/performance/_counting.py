"""Deterministic store-operation counting.

The performance gate asserts on *counts*, not seconds: ngio's performance
regressions are algorithmic (metadata re-parsed N times, one group opened per
well) and those show up as exact integers with zero variance. Counts are also
backend-independent, so a local-store measurement predicts what an operation
costs on S3, where every op is a network round-trip.

`CountingNgioStore` subclasses `NgioStore` rather than wrapping it. Wrapping is
not an option: `NgioStore.store_type`, `full_url`, `local_root`, `memory_dict`
and `sync_fs_and_path` all `isinstance`-check `self._store`, so an extra layer
in between degrades the store to `"other"` and breaks the table backends.
Subclassing survives ngio's normalisation because `NgioStore.from_any` returns
an existing `NgioStore` unchanged and `_with_store` rebuilds via `type(self)`.
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from zarr.abc.store import Store

from ngio.utils import NgioStore

if TYPE_CHECKING:
    import builtins
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Iterable,
        Iterator,
    )

    from zarr.abc.buffer import Buffer
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype

__all__ = [
    "COUNTER_KEYS",
    "CountingNgioStore",
    "assert_instrumentation_complete",
    "count",
    "record",
    "zero_fill",
]

# Filenames that carry zarr/OME metadata rather than array data. Splitting ops
# on this is what makes "12 metadata reads to open one image" legible.
_META_TAILS = frozenset({"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"})

#: Every counter the gate emits, always present and zero-filled so a baseline
#: diff shows a changed value rather than an added key.
COUNTER_KEYS: tuple[str, ...] = (
    "bytes.read",
    "bytes.written",
    "clear",
    "delete",
    "delete_dir",
    "exists",
    "get.chunk",
    "get.meta",
    "get.partial",
    "get_many",
    "get_partial_values",
    "getsize",
    "getsize_prefix",
    "is_empty",
    "list",
    "list_dir",
    "list_prefix",
    "set.chunk",
    "set.meta",
    "set_if_not_exists",
    "set_many",
    # Recorded by `tests.performance._probes`, not by the store. These measure work
    # that never touches the store at all — chiefly pydantic re-validation —
    # which store counters are structurally blind to.
    "image.dimensions",
    "meta.attrs_load",
    "meta.decode_attempt",
    "meta.decode_call",
    "meta.decode_fail",
    "meta.group_reopen",
    "plate.get_well",
)

_LOCK = threading.Lock()
_ACTIVE: Counter[str] | None = None


def _bump(name: str, n: int = 1) -> None:
    """Add `n` to counter `name` if a recording block is active."""
    if _ACTIVE is None:
        return
    with _LOCK:
        _ACTIVE[name] += n


def record(name: str, n: int = 1) -> None:
    """Add `n` to counter `name`, for instrumentation outside the store.

    Used by `tests.performance._probes` so that probe counters and store counters land
    in the same tally and appear side by side in one baseline entry.
    """
    _bump(name, n)


def _kind(key: str) -> str:
    """Classify a store key as metadata or chunk data."""
    return "meta" if key.rsplit("/", 1)[-1] in _META_TAILS else "chunk"


@contextmanager
def count() -> Iterator[Counter[str]]:
    """Record every store operation performed inside the block.

    The recorder is a module global rather than instance state so that stores
    pickled into dask graphs carry no counter with them, and so that ops issued
    by dask worker threads land in the same tally.

    Raises:
        RuntimeError: If a block is already active. Nested blocks would
            silently attribute inner ops to both, so they are refused.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("count() blocks cannot be nested")
    recorder: Counter[str] = Counter()
    _ACTIVE = recorder
    try:
        yield recorder
    finally:
        _ACTIVE = None


def zero_fill(counters: Counter[str]) -> dict[str, int]:
    """Expand a tally to the full key set, sorted, for stable baseline diffs."""
    return {key: int(counters.get(key, 0)) for key in COUNTER_KEYS}


class CountingNgioStore(NgioStore):
    """An `NgioStore` that tallies each store call into the active recorder.

    Only the async surface is overridden. zarr 3.1 and 3.2 expose no sync IO
    methods on the `Store` ABC, and the `_get_bytes`/`_get_json` helpers
    delegate to `get`, so they are counted transitively.
    """

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        _bump(f"get.{_kind(key)}")
        if byte_range is not None:
            _bump("get.partial")
        buffer = await super().get(key, prototype, byte_range)
        if buffer is not None:
            _bump("bytes.read", len(buffer))
        return buffer

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> builtins.list[Buffer | None]:
        key_ranges = list(key_ranges)
        _bump("get_partial_values", len(key_ranges))
        buffers = await super().get_partial_values(prototype, key_ranges)
        _bump("bytes.read", sum(len(b) for b in buffers if b is not None))
        return buffers

    async def set(self, key: str, value: Buffer) -> None:
        _bump(f"set.{_kind(key)}")
        _bump("bytes.written", len(value))
        return await super().set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        _bump("set_if_not_exists")
        return await super().set_if_not_exists(key, value)

    async def _get_many(
        self,
        requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]],
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        # Counts batch invocations; the individual reads are counted by `get`,
        # which the parent implementation routes through.
        _bump("get_many")
        async for item in super()._get_many(requests):
            yield item

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        _bump("set_many")
        return await super()._set_many(values)

    async def exists(self, key: str) -> bool:
        _bump("exists")
        return await super().exists(key)

    async def delete(self, key: str) -> None:
        _bump("delete")
        return await super().delete(key)

    async def delete_dir(self, prefix: str) -> None:
        _bump("delete_dir")
        return await super().delete_dir(prefix)

    async def is_empty(self, prefix: str) -> bool:
        _bump("is_empty")
        return await super().is_empty(prefix)

    async def clear(self) -> None:
        _bump("clear")
        return await super().clear()

    async def getsize(self, key: str) -> int:
        _bump("getsize")
        return await super().getsize(key)

    async def getsize_prefix(self, prefix: str) -> int:
        _bump("getsize_prefix")
        return await super().getsize_prefix(prefix)

    def list(self) -> AsyncIterator[str]:
        _bump("list")
        return super().list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        _bump("list_prefix")
        return super().list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        _bump("list_dir")
        return super().list_dir(prefix)


#: Store methods this module hooks explicitly.
_COVERED = frozenset(
    {
        "_get_many",
        "_set_many",
        "clear",
        "delete",
        "delete_dir",
        "exists",
        "get",
        "get_partial_values",
        "getsize",
        "getsize_prefix",
        "is_empty",
        "list",
        "list_dir",
        "list_prefix",
        "set",
        "set_if_not_exists",
    }
)

#: Store members that perform no IO of their own, or reach the store only via a
#: method in `_COVERED`.
_NON_IO = frozenset(
    {
        "_abc_impl",
        "_check_writable",
        "_ensure_open",
        "_get_bytes",  # delegates to get
        "_get_bytes_sync",  # delegates to get
        "_get_json",  # delegates to get
        "_get_json_sync",  # delegates to get
        "_open",
        "close",
        "open",
        "read_only",
        "supports_consolidated_metadata",
        "supports_deletes",
        "supports_listing",
        "supports_partial_writes",
        "supports_writes",
        "with_read_only",
    }
)


def assert_instrumentation_complete() -> None:
    """Fail if zarr's `Store` grew an IO method this module does not count.

    Without this a zarr upgrade would silently drop operations from the tally,
    which reads as a performance *improvement* in the baseline diff.

    Raises:
        AssertionError: If an unclassified `Store` member is found.
    """
    members: set[str] = {
        name
        for name in dir(Store)
        if not name.startswith("__") and callable(getattr(Store, name, None))
    }
    unclassified = members - _COVERED - _NON_IO
    if unclassified:
        raise AssertionError(
            "zarr's Store has members this module does not classify: "
            f"{sorted(unclassified)}. Hook them in CountingNgioStore (and add "
            "them to COUNTER_KEYS) if they perform IO, otherwise list them in "
            "_NON_IO."
        )


def counting_store(source: Any) -> CountingNgioStore:
    """Build a `CountingNgioStore` from any store-like object.

    `NgioStore.from_any` constructs through `cls`, so this returns the counting
    subclass, and ngio's own normalisation then passes it through untouched.
    """
    return CountingNgioStore.from_any(source)
