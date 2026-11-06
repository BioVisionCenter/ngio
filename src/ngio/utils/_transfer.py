"""Copy whole OME-Zarr hierarchies between stores.

`to_memory`/`from_memory` move a hierarchy in and out of RAM; `to_ozx`/`from_ozx`
pack and unpack an OME-Zarr zip file (RFC-9). Together they are the explicit
"extract, modify, repack" workflow the RFC recommends for heavy or parallel
writes, where editing the archive in place is the wrong tool.
"""

from __future__ import annotations

import os
from pathlib import Path

from zarr.storage import MemoryStore

from ngio.utils._errors import NgioValueError
from ngio.utils._ozx import OZX_SUFFIX, OzxStore
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    StoreOrGroup,
    copy_group,
    open_group_wrapper,
)


def _copy_hierarchy(src: StoreOrGroup, dst: StoreOrGroup, *, overwrite: bool) -> None:
    mode: AccessModeLiteral = "w" if overwrite else "w-"
    # A `.ozx` path becomes a store this function opens and therefore closes;
    # a store object stays the caller's to close.
    owned: OzxStore | None = None
    if isinstance(dst, str | Path) and os.fspath(dst).endswith(OZX_SUFFIX):
        if "://" in os.fspath(dst):
            # zipfile needs a seekable local file; a remote archive would have to
            # be packed locally and uploaded, which is not offered yet.
            raise NgioValueError(
                f"An OME-Zarr zip file can only be written to a local path ({dst})."
            )
        owned = OzxStore(dst, mode=mode)
    try:
        src_group = open_group_wrapper(src, mode="r")
        dst_group = open_group_wrapper(
            owned if owned is not None else dst,
            mode=mode,
            zarr_format=src_group.metadata.zarr_format,
            attributes=src_group.attrs.asdict(),
        )
        # The copy warns when it cannot take the filesystem shortcut; a memory
        # or zip endpoint never can, so the warning carries no information here.
        copy_group(src_group, dst_group, suppress_warnings=True)
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


def to_ozx(src: StoreOrGroup, path: Path | str, *, overwrite: bool = False) -> Path:
    """Pack an OME-Zarr hierarchy into an OME-Zarr zip file (RFC-9).

    The archive is written once, closed and conformant on return: no dead
    space, `zarr.json` records first, archive comment set. The source must be
    OME-Zarr 0.5 (Zarr v3); RFC-9 has no Zarr v2 form.

    Args:
        src: The hierarchy to pack (path, store or group).
        path: The `.ozx` file to create.
        overwrite: Replace an existing file.

    Returns:
        The path of the archive.
    """
    _copy_hierarchy(src, path, overwrite=overwrite)
    return Path(path)


def from_ozx(path: Path | str, dst: StoreOrGroup, *, overwrite: bool = False) -> None:
    """Unpack an OME-Zarr zip file into `dst` (a path, store or group).

    Args:
        path: The `.ozx` file to read.
        dst: Where to write the hierarchy.
        overwrite: Replace an existing hierarchy at `dst`.
    """
    _copy_hierarchy(OzxStore(path, mode="r"), dst, overwrite=overwrite)
