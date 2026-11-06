"""Copy whole OME-Zarr hierarchies between stores.

`to_memory`/`from_memory` move a hierarchy in and out of RAM; `to_ozx`/`from_ozx`
pack and unpack an OME-Zarr zip file (RFC-9). Together they are the explicit
"extract, modify, repack" workflow the RFC recommends for heavy or parallel
writes, where editing the archive in place is the wrong tool.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import fsspec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from ngio.utils._errors import NgioFileExistsError
from ngio.utils._ozx import OZX_SUFFIX, OzxStore, is_remote, open_ozx
from ngio.utils._store import NgioStore
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    StoreOrGroup,
    open_group_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import zarr


def _is_ozx_location(dst: object) -> bool:
    return isinstance(dst, str | Path) and os.fspath(dst).endswith(OZX_SUFFIX)


def _upload_ozx(src: StoreOrGroup, url: str, *, overwrite: bool) -> None:
    """Pack into a local temporary file, then upload: zip needs a seekable file."""
    fs, remote_path = fsspec.core.url_to_fs(url)
    if not overwrite and fs.exists(remote_path):
        raise NgioFileExistsError(f"{url} already exists; pass overwrite=True.")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(remote_path).name
        _copy_hierarchy(src, local, overwrite=True)
        fs.put_file(os.fspath(local), remote_path)


# Written when the destination group is created, from the source's attributes.
_ROOT_METADATA = frozenset({"zarr.json", ".zgroup", ".zattrs", ".zmetadata"})


def _copy_keys(src_group: zarr.Group, dst_group: zarr.Group) -> None:
    """Copy every key under the source group, byte for byte.

    Key level rather than node level: a zarr walk misses the files that
    table backends keep beside the zarr nodes (parquet, csv). The source's
    root metadata document is skipped: the destination group was created with
    the same format and attributes, and ngio neither writes nor trusts
    consolidated metadata.
    """
    src_store, dst_store = src_group.store, dst_group.store
    src_prefix = f"{src_group.path}/" if src_group.path else ""
    dst_prefix = f"{dst_group.path}/" if dst_group.path else ""
    prototype = default_buffer_prototype()

    async def copy_one(key: str) -> None:
        rel = key[len(src_prefix) :]
        if rel in _ROOT_METADATA:
            return
        value = await src_store.get(key, prototype=prototype)
        if value is not None:
            await dst_store.set(dst_prefix + rel, value)

    async def run() -> None:
        pending: list[Awaitable[None]] = []
        async for key in src_store.list_prefix(src_prefix):
            pending.append(copy_one(key))
            if len(pending) >= _COPY_BATCH:
                await asyncio.gather(*pending)
                pending = []
        await asyncio.gather(*pending)

    sync(run())
    # The copy ends on chunk writes, which a single-file store publishes only
    # when its directory is written.
    NgioStore.ensure(dst_store).flush()


_COPY_BATCH = 32


def _copy_hierarchy(src: StoreOrGroup, dst: StoreOrGroup, *, overwrite: bool) -> None:
    mode: AccessModeLiteral = "w" if overwrite else "w-"
    # A `.ozx` path becomes a store this function opens and therefore closes;
    # a store object stays the caller's to close.
    owned: OzxStore | None = None
    if _is_ozx_location(dst):
        if is_remote(cast("str | Path", dst)):
            _upload_ozx(src, os.fspath(cast("str | Path", dst)), overwrite=overwrite)
            return
        owned = OzxStore(cast("str | Path", dst), mode=mode)
    try:
        src_group = open_group_wrapper(src, mode="r")
        dst_group = open_group_wrapper(
            owned if owned is not None else dst,
            mode=mode,
            zarr_format=src_group.metadata.zarr_format,
            attributes=src_group.attrs.asdict(),
        )
        _copy_keys(src_group, dst_group)
    finally:
        if owned is not None:
            owned.close()


def to_memory(src: StoreOrGroup) -> MemoryStore:
    """Copy an OME-Zarr hierarchy into a new in-memory store.

    The whole hierarchy is loaded into RAM. Open the result like any store:
    `open_ome_zarr_container(to_memory("image.ozx"))`.
    """
    store = MemoryStore()
    _copy_hierarchy(src, store, overwrite=False)
    return store


def from_memory(
    src: MemoryStore | dict, dst: StoreOrGroup, *, overwrite: bool = False
) -> None:
    """Copy an in-memory OME-Zarr hierarchy to `dst` (a path, store or group).

    Args:
        src: The in-memory store, or the dict backing one.
        dst: Where to write; a `.ozx` path produces an OME-Zarr zip file.
        overwrite: Replace an existing hierarchy at `dst`.
    """
    _copy_hierarchy(src, dst, overwrite=overwrite)


def to_ozx(
    src: StoreOrGroup, path: Path | str, *, overwrite: bool = False
) -> Path | str:
    """Pack an OME-Zarr hierarchy into an OME-Zarr zip file (RFC-9).

    The archive is written once, closed and conformant on return: no dead
    space, `zarr.json` records first, archive comment set. The source must be
    OME-Zarr 0.5 (Zarr v3); RFC-9 has no Zarr v2 form. A URL target is packed
    into a local temporary file and uploaded.

    Args:
        src: The hierarchy to pack (path, store or group).
        path: The `.ozx` file to create, a local path or a URL.
        overwrite: Replace an existing file.

    Returns:
        The archive location: a `Path` for a local file, the URL otherwise.
    """
    _copy_hierarchy(src, path, overwrite=overwrite)
    return path if is_remote(path) else Path(path)


def from_ozx(path: Path | str, dst: StoreOrGroup, *, overwrite: bool = False) -> None:
    """Unpack an OME-Zarr zip file into `dst` (a path, store or group).

    Args:
        path: The `.ozx` file to read, a local path or a URL.
        dst: Where to write the hierarchy.
        overwrite: Replace an existing hierarchy at `dst`.
    """
    _copy_hierarchy(open_ozx(path, mode="r"), dst, overwrite=overwrite)
