"""Copy whole OME-Zarr hierarchies between stores.

`to_memory`/`from_memory` move a hierarchy in and out of RAM; `to_ozx`/`from_ozx`
pack and unpack an OME-Zarr zip file (RFC-9). Together they are the explicit
"extract, modify, repack" workflow the RFC recommends for heavy or parallel
writes, where editing the archive in place is the wrong tool.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

import fsspec
import zarr
from zarr.abc.store import Store
from zarr.storage import MemoryStore

from ngio.utils._errors import NgioFileExistsError, NgioValueError
from ngio.utils._ozx import OZX_SUFFIX, OzxStore, is_remote, live_writer, open_ozx
from ngio.utils._store import NgioStore
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    StoreOrGroup,
    copy_keys,
    open_group_wrapper,
)


def _is_ozx_location(dst: object) -> bool:
    return isinstance(dst, str | Path) and os.fspath(dst).endswith(OZX_SUFFIX)


def _upload_ozx(src: StoreOrGroup, url: str, *, overwrite: bool) -> None:
    """Pack into a local temporary file, then upload: zip needs a seekable file."""
    fs, remote_path = fsspec.core.url_to_fs(url)
    if not overwrite and fs.exists(remote_path):
        raise NgioFileExistsError(f"{url} already exists; pass overwrite=True.")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(remote_path).name
        _copy_hierarchy(src, local, overwrite=True, archive=True)
        fs.put_file(os.fspath(local), remote_path)


def _location_key(target: object) -> str | None:
    """`NgioStore.location_key` for a store, a group or a path/URL argument."""
    if isinstance(target, zarr.Group):
        target = target.store
    if isinstance(target, str | Path):
        if is_remote(target):
            return os.fspath(target).rstrip("/")
        return os.fspath(Path(target).resolve())
    if isinstance(target, NgioStore | Store):
        return NgioStore.ensure(target).location_key
    return None


def _refuse_same_location(src_group: zarr.Group, dst: StoreOrGroup) -> None:
    """Opening the destination with `mode="w"` clears it: never the source."""
    src_key, dst_key = _location_key(src_group), _location_key(dst)
    if src_key is not None and src_key == dst_key:
        raise NgioValueError(
            "The source and the destination of the copy are the same location "
            f"({dst_key}); copying would clear the source first. Copy into a "
            "new file, or `OzxStore.repack()` to reclaim dead space in an archive."
        )


def _copy_hierarchy(
    src: StoreOrGroup,
    dst: StoreOrGroup,
    *,
    overwrite: bool,
    archive: bool | None = None,
) -> None:
    """Copy `src` into `dst`; `archive` forces (or forbids) a zip destination.

    `None` decides by the `.ozx` suffix, which is how a path argument is routed
    everywhere else in ngio. `to_ozx` passes `True`: its destination is an
    archive whatever the name, so a `.zip` target never becomes a directory.
    """
    mode: AccessModeLiteral = "w" if overwrite else "w-"
    if archive is None:
        archive = _is_ozx_location(dst)
    elif archive and not isinstance(dst, str | Path):
        raise NgioValueError(
            f"An OME-Zarr zip file is written to a path or URL, got {type(dst)}."
        )
    # Open the source before anything touches the destination: `mode="w"`
    # clears it, and a `.ozx` destination truncates the file as it opens.
    src_group = open_group_wrapper(src, mode="r")
    _refuse_same_location(src_group, dst)
    # A `.ozx` path becomes a store this function opens and therefore closes;
    # a store object, or the live writer of an archive already open in this
    # process, stays the caller's to close.
    owned: OzxStore | None = None
    if archive:
        location = cast("str | Path", dst)
        if is_remote(location):
            _upload_ozx(src, os.fspath(location), overwrite=overwrite)
            return
        if live_writer(location) is None:
            owned = OzxStore(location, mode=mode)
        else:
            dst = open_ozx(location, mode=mode)
    try:
        dst_group = open_group_wrapper(
            owned if owned is not None else dst,
            mode=mode,
            zarr_format=src_group.metadata.zarr_format,
            attributes=src_group.attrs.asdict(),
        )
        copy_keys(src_group, dst_group)
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
    into a local temporary file and uploaded. The file is an archive whatever
    its name; a name not ending in `.ozx` warns, as the RFC recommends the
    suffix.

    Args:
        src: The hierarchy to pack (path, store or group).
        path: The `.ozx` file to create, a local path or a URL.
        overwrite: Replace an existing file.

    Returns:
        The archive location: a `Path` for a local file, the URL otherwise.
    """
    _copy_hierarchy(src, path, overwrite=overwrite, archive=True)
    return path if is_remote(path) else Path(path)


def from_ozx(path: Path | str, dst: StoreOrGroup, *, overwrite: bool = False) -> None:
    """Unpack an OME-Zarr zip file into `dst` (a path, store or group).

    Args:
        path: The `.ozx` file to read, a local path or a URL.
        dst: Where to write the hierarchy.
        overwrite: Replace an existing hierarchy at `dst`.
    """
    _copy_hierarchy(open_ozx(path, mode="r"), dst, overwrite=overwrite)
