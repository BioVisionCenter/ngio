"""Common utilities for working with Zarr groups in consistent ways."""

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import quote

import dask.array as da
import fsspec
import zarr
from filelock import BaseFileLock, FileLock
from pydantic_zarr.v2 import ArraySpec as AnyArraySpecV2
from pydantic_zarr.v3 import ArraySpec as AnyArraySpecV3
from zarr.abc.store import Store
from zarr.errors import ContainsGroupError
from zarr.storage import FsspecStore, LocalStore, MemoryStore, ZipStore

from ngio.config import NgioConfig, get_config
from ngio.utils import _retry
from ngio.utils._cache import NgioCache
from ngio.utils._codec_pipeline import warn_on_codec_pipeline_fallback
from ngio.utils._errors import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioValueError,
)
from ngio.utils._store import NgioStore
from ngio.utils._warnings import NgioUserWarning

logger = logging.getLogger(f"ngio:{__name__}")

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
# StoreLike is more restrictive than it could be
# but to make sure we can handle the store correctly
# we need to be more restrictive
NgioSupportedStore: TypeAlias = (
    str
    | Path
    | fsspec.mapping.FSMap
    | FsspecStore
    | MemoryStore
    | dict
    | LocalStore
    | ZipStore
    | NgioStore
)
GenericStore: TypeAlias = NgioSupportedStore | Store
StoreOrGroup: TypeAlias = NgioSupportedStore | zarr.Group


def _check_group(
    group: zarr.Group, mode: AccessModeLiteral | None = None
) -> zarr.Group:
    """Check the group and return a valid group backed by an NgioStore."""
    if group.read_only and mode not in [None, "r"]:
        raise NgioValueError(f"The group is read only. Cannot open in mode {mode}.")

    needs_read_only_reopen = mode == "r" and not group.read_only
    # Not `isinstance(..., NgioStore)`: with the default retry policy a plain
    # local store is *already* the normalized form, and reopening it here would
    # cost a full metadata read on every `open_group_wrapper(group)` call.
    if not NgioStore.is_normalized(group.store) or needs_read_only_reopen:
        ngio_store = NgioStore.normalize(group.store)
        reopen_mode = "r" if (mode == "r" or group.read_only) else "r+"
        group = zarr.open_group(
            store=ngio_store,
            path=group.path,
            mode=reopen_mode,
            zarr_format=group.metadata.zarr_format,
            use_consolidated=False,
        )
    return group


def open_group_wrapper(
    store: StoreOrGroup,
    mode: AccessModeLiteral | None = None,
    zarr_format: Literal[2, 3] | None = None,
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    The group is opened on an `NgioStore` whenever that wrapper changes
    behaviour — a configured `io_retry`, or Windows, where the sharing-violation
    retry is unconditional — so those store IO calls (including dask-worker
    chunk reads/writes) honor the policy. With the default policy on a plain
    local store the bare store is handed to zarr instead: `NgioStore` is a
    `WrapperStore`, and zarr silently falls back to its own codec pipeline for
    stores a third-party pipeline does not recognise, so wrapping a pass-through
    would cost `zarrs` for nothing. `ZarrGroupHandler.store` exposes the
    `NgioStore` services either way.

    Consolidated metadata is never used, here or in `reopen_group`. ngio does
    not write it and cannot refresh someone else's: zarr leaves `.zmetadata`
    untouched when attributes are written, so trusting it makes ngio read back
    a snapshot from before its own writes — a label ngio had just created was
    invisible to the next `list_labels()`. Ignoring it also drops a `.zmetadata`
    probe that misses on every ngio-written store, which is a wasted round-trip
    per metadata read remotely.

    Args:
        store (StoreOrGroup): The store or group to open.
        mode (AccessModeLiteral): The mode to open the group in.
        zarr_format (int): The Zarr format version to use.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        return _check_group(store, mode)

    try:
        mode = mode if mode is not None else "a"
        ngio_store = NgioStore.normalize(store, mode=mode)
        group = zarr.open_group(
            store=ngio_store,
            mode=mode,
            zarr_format=zarr_format,
            use_consolidated=False,
        )

    except FileExistsError as e:
        raise NgioFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    except FileNotFoundError as e:
        raise NgioFileNotFoundError(f"No Zarr group found at {store}") from e

    except ContainsGroupError as e:
        raise NgioFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    return group


class ZarrGroupHandler:
    """A simple wrapper around a Zarr group to handle metadata."""

    def __init__(
        self,
        store: StoreOrGroup,
        zarr_format: Literal[2, 3] | None = None,
        cache: bool = False,
        mode: AccessModeLiteral | None = None,
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            zarr_format (int | None): The Zarr format version to use.
            cache (bool): Whether to cache the metadata.
            mode (str | None): The mode of the store.
        """
        if mode not in ["r", "r+", "w", "w-", "a", None]:
            raise NgioValueError(f"Mode {mode} is not supported.")

        group = open_group_wrapper(store=store, mode=mode, zarr_format=zarr_format)
        self._group = group
        self.use_cache = cache

        self._group_cache: NgioCache[zarr.Group] = NgioCache(use_cache=cache)
        self._array_cache: NgioCache[zarr.Array] = NgioCache(use_cache=cache)
        self._handlers_cache: NgioCache[ZarrGroupHandler] = NgioCache(use_cache=cache)
        # The group's own attributes, which is where every NGFF document lives.
        # Held separately from `_group_cache` (which holds *children*) because
        # `load_attrs` is the one choke point every metadata read passes
        # through, so this is the only place a store round-trip can be saved.
        self._attrs_cache: dict | None = None
        self._lock: tuple[Path, BaseFileLock] | None = None
        # `_group.store` never changes for a handler: `reopen_group` passes the
        # same object straight through.
        self._store_facade: NgioStore | None = None

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"ZarrGroupHandler(full_url={self.full_url}, read_only={self.read_only}, "
            f"cache={self.use_cache}"
        )

    @property
    def store(self) -> NgioStore:
        """Return the store of the group, as an `NgioStore` facade.

        The group itself may hold a bare store (see `open_group_wrapper`), so
        this can be a wrapper built purely to expose `full_url`, `get_mapper`
        and friends. Built once: `NgioStore.__init__` deep-copies the retry
        config, and `full_url`, `_create_lock` and both table backends read
        this per call.
        """
        if self._store_facade is None:
            self._store_facade = NgioStore.ensure(self._group.store)
        return self._store_facade

    @property
    def path(self) -> str:
        """Return the group's path inside the store.

        Reads `_group` rather than the `group` property on purpose: the path is
        fixed at construction and `reopen_group` carries it through unchanged,
        while `group` reopens the group -- a full metadata read -- when caching
        is off. `_create_lock` avoids the same trap the same way.
        """
        return self._group.path

    @property
    def full_url(self) -> str | None:
        """Return the store path."""
        return self.store.full_url(self.path)

    @property
    def zarr_format(self) -> Literal[2, 3]:
        """Return the Zarr format version."""
        return self._group.metadata.zarr_format

    @property
    def read_only(self) -> bool:
        """Return whether the group is read only."""
        return self._group.read_only

    def _create_lock(self) -> tuple[Path, BaseFileLock]:
        """Create the lock."""
        if self._lock is not None:
            return self._lock

        # Caching used to be refused here. It is compatible now: `locked()`
        # invalidates on entry and exit, so a read-modify-write under the lock
        # cannot be served a value cached from before the lock was taken.
        local_root = self.store.local_root
        if local_root is None:
            raise NgioValueError(
                "The store needs to be a LocalStore to use the lock mechanism. "
                f"Instead, got a {self.store.store_type} store."
            )

        # After the guard above, not before: under `-W error` a warning raised
        # first would mask it, so a remote store on Windows would fail with
        # `NgioUserWarning` where every other platform raises `NgioValueError`.
        if _retry._IS_WINDOWS:
            # Warned rather than refused: an *uncontended* lock is exclusive on
            # Windows too, so a single writer is correct and refusing would
            # break it for no reason. Only concurrent writers are at risk.
            warnings.warn(
                "ngio's file lock is not exclusive on Windows: `filelock` "
                "deletes the lock file on release without the descriptor check "
                "its Unix backend has (`_windows.py::_acquire` has no "
                "`st_nlink` guard), so two concurrent writers can hold it at "
                "once and one update can be lost. Proceeding with a best-effort "
                "lock, which is correct for a single writer. Run concurrent "
                "writers on Linux or macOS.",
                NgioUserWarning,
                stacklevel=2,
            )

        # Locks live in a directory beside the store, never inside it: a lock
        # file under `plate.zarr/` would be a non-Zarr entry every reader has to
        # tolerate, and it would travel with the data on copy. The group path is
        # percent-encoded rather than used as-is, so `C/03` stays one flat name
        # and two groups differing only after a dot (`foo.bar` vs `foo.baz`)
        # cannot collide onto one lock — `with_suffix` mapped both to `foo.lock`.
        #
        # `self._group.path`, not `self.group.path`: the latter reopens the
        # group (a full metadata read) when caching is off, and the path is
        # fixed at construction — `reopen_group` passes it through unchanged.
        locks_root = local_root.with_name(f"{local_root.name}.ngio-locks")
        # Created here rather than left to `filelock`, which only grew that
        # behaviour in 3.12.3 while ngio's floor is 3.12. `exist_ok` because
        # concurrent workers reach this at the same time.
        locks_root.mkdir(parents=True, exist_ok=True)
        _lock_path = locks_root / f"{quote(self._group.path, safe='') or 'root'}.lock"
        _lock = FileLock(_lock_path, timeout=10)
        return _lock_path, _lock

    @property
    def lock(self) -> BaseFileLock:
        """Return the lock."""
        if self._lock is None:
            self._lock = self._create_lock()
        return self._lock[1]

    @property
    def lock_path(self) -> Path:
        """Return the lock path."""
        if self._lock is None:
            self._lock = self._create_lock()
        return self._lock[0]

    def reopen_group(self) -> zarr.Group:
        """Reopen the group.

        This is useful when the group has been modified
        outside of the handler.
        """
        mode = "r" if self.read_only else "r+"
        return zarr.open_group(
            store=self._group.store,
            path=self._group.path,
            mode=mode,
            zarr_format=self._group.metadata.zarr_format,
            use_consolidated=False,
        )

    def reopen_handler(self) -> "ZarrGroupHandler":
        """Reopen the handler.

        This is useful when the group has been modified
        outside of the handler.
        """
        mode = "r" if self.read_only else "r+"
        group = self.reopen_group()
        return ZarrGroupHandler(
            store=group,
            zarr_format=group.metadata.zarr_format,
            cache=self.use_cache,
            mode=mode,
        )

    def invalidate_meta(self) -> None:
        """Drop cached metadata, keeping child handlers and the lock alive.

        Narrower than `clean_cache`: it refreshes what a metadata read sees and
        cascades to child handlers, but does not discard them, and never touches
        `_lock`. Both matter — `clean_cache` runs inside a held lock in the HCS
        write paths, where discarding every open well handler per added image
        would be gratuitous, and re-creating the `FileLock` would hand back a
        different object for a path this thread already holds.
        """
        self._attrs_cache = None
        if self.use_cache:
            # Only when caching. With `cache=False` nothing is held: `_group` is
            # never read for its attributes (`load_attrs` reopens every call)
            # and the three caches are disabled and empty, so reopening here
            # would be a pure extra round-trip on every write.
            self._group = self.reopen_group()
        self._group_cache.clear()
        self._array_cache.clear()
        for handler in self._handlers_cache.cache.values():
            handler.invalidate_meta()

    def clean_cache(self) -> None:
        """Clear every cached object, so the next access reads the store."""
        self.invalidate_meta()
        self._handlers_cache.clear()

    def refresh(self) -> None:
        """Re-read metadata that `cache=True` would otherwise hold indefinitely.

        The answer to "someone else wrote, and I want to see it". A no-op with
        `cache=False`, where nothing is held in the first place.
        """
        self.invalidate_meta()

    @contextmanager
    def locked(self) -> Iterator[BaseFileLock]:
        """Hold the group's file lock, with cached metadata refreshed around it.

        Invalidating on the way in is what makes a read-modify-write under the
        lock correct when caching is on: without it, the value read inside the
        lock could have been cached before the lock was taken, and the write
        would silently drop whatever the previous holder did. Invalidating on
        the way out keeps a later unlocked read from seeing the pre-write state.

        This is why caching and locking are compatible at all — before it, the
        two were mutually exclusive by an outright refusal in `_create_lock`.
        """
        with self.lock:
            # Inside the lock, never before it. Refreshing on the way to the
            # lock snapshots a state the previous holder is still editing, and
            # the read below would then return it — which is precisely the lost
            # update this is here to prevent.
            self.invalidate_meta()
            try:
                yield self.lock
            finally:
                self.invalidate_meta()

    @property
    def group(self) -> zarr.Group:
        """Return the group."""
        if self.use_cache is False:
            # If we are not using cache, we need to reopen the group
            # to make sure that the attributes are up to date
            return self.reopen_group()
        return self._group

    def load_attrs(self) -> dict:
        """Load the attributes of the group.

        With `cache=False` this reads the store every call, which is what makes
        ngio's metadata unconditionally fresh. With `cache=True` it is served
        from memory until something invalidates it: a write through this
        handler, `refresh()`, or taking the group's lock.

        A copy is returned, not the cached dict, so a caller that edits what it
        was handed cannot corrupt the next reader.
        """
        if self._attrs_cache is not None:
            return deepcopy(self._attrs_cache)
        if self.use_cache:
            # `self._group`, not a reopen: it is current by construction and
            # refreshed by `invalidate_meta`, and its attributes are already in
            # memory — so the first read of a cached handler costs nothing
            # either, not just the repeats.
            attrs = self._group.attrs.asdict()
            self._attrs_cache = attrs
            return deepcopy(attrs)
        return self.reopen_group().attrs.asdict()

    def write_attrs(self, attrs: dict, overwrite: bool = False) -> None:
        """Write the metadata to the store."""
        # Maybe we should use the lock here
        if self.read_only:
            raise NgioValueError("The group is read only. Cannot write metadata.")
        group = self.reopen_group()
        if overwrite:
            group.attrs.clear()
        group.attrs.update(attrs)
        # The write went through a freshly opened group, so `_group` and the
        # attrs cache both describe the state from before it.
        self.invalidate_meta()

    def create_group(self, path: str, overwrite: bool = False) -> zarr.Group:
        """Create a group in the group."""
        if self.group.read_only:
            raise NgioValueError("Cannot create a group in read only mode.")

        try:
            group = self.group.create_group(path, overwrite=overwrite)
        except ContainsGroupError as e:
            raise NgioFileExistsError(
                f"A Zarr group already exists at {path}, "
                "consider setting overwrite=True."
            ) from e
        self._group_cache.set(path, group, overwrite=overwrite)
        return group

    def get_group(
        self,
        path: str,
        create_mode: bool = False,
        overwrite: bool = False,
    ) -> zarr.Group:
        """Get a group from the group.

        Args:
            path (str): The path to the group.
            create_mode (bool): If True, create the group if it does not exist.
            overwrite (bool): If True, overwrite the group if it exists.

        Returns:
            zarr.Group: The Zarr group.

        """
        if overwrite and not create_mode:
            raise NgioValueError("Cannot overwrite a group without create_mode=True.")

        if overwrite:
            return self.create_group(path, overwrite=overwrite)

        group = self._group_cache.get(path)
        if isinstance(group, zarr.Group):
            return group

        group = self.group.get(path, default=None)
        if isinstance(group, zarr.Group):
            self._group_cache.set(path, group, overwrite=overwrite)
            return group

        if isinstance(group, zarr.Array):
            raise NgioValueError(f"The object at {path} is not a group, but an array.")

        if not create_mode:
            raise NgioFileNotFoundError(f"No group found at {path}")

        try:
            group = self.create_group(path)
        except NgioFileExistsError:
            # Another worker created the group between the lookup above and
            # this call. `create_mode=True` means get-or-create, so losing that
            # race is not an error: adopt the group the winner wrote. Without
            # this, two workers adding images to the same well would race on
            # creating it and one would fail outright.
            group = self.group.get(path, default=None)
            if not isinstance(group, zarr.Group):
                raise
        self._group_cache.set(path, group, overwrite=overwrite)
        return group

    def get_array(self, path: str) -> zarr.Array:
        """Get an array from the group."""
        array = self._array_cache.get(path)
        if isinstance(array, zarr.Array):
            return array
        array = self.group.get(path, default=None)
        if isinstance(array, zarr.Array):
            # On the fetch, not on a cache hit: one array per pyramid level per
            # image, and the warning itself is deduplicated downstream.
            warn_on_codec_pipeline_fallback(array)
            self._array_cache.set(path, array)
            return array

        if isinstance(array, zarr.Group):
            raise NgioValueError(f"The object at {path} is not an array, but a group.")
        raise NgioFileNotFoundError(f"No array found at {path}")

    def get_handler(
        self,
        path: str,
        create_mode: bool = True,
        overwrite: bool = False,
    ) -> "ZarrGroupHandler":
        """Get a new handler for a group in the current handler group.

        Args:
            path (str): The path to the group.
            create_mode (bool): If True, create the group if it does not exist.
            overwrite (bool): If True, overwrite the group if it exists.
        """
        handler = self._handlers_cache.get(path)
        if handler is not None:
            return handler
        group = self.get_group(path, create_mode=create_mode, overwrite=overwrite)
        mode = "r" if group.read_only else "r+"
        handler = ZarrGroupHandler(
            store=group, zarr_format=self.zarr_format, cache=self.use_cache, mode=mode
        )
        self._handlers_cache.set(path, handler)
        return handler

    @property
    def is_listable(self) -> bool:
        return is_group_listable(self.group)

    def delete_group(self, path: str) -> None:
        """Delete a group from the current group.

        Args:
            path (str): The path to the group to delete.
        """
        if self.group.read_only:
            raise NgioValueError("Cannot delete a group in read only mode.")
        self.group.__delitem__(path)
        self._group_cache._cache.pop(path, None)
        self._handlers_cache._cache.pop(path, None)

    def delete_self(self) -> None:
        """Delete the current group."""
        if self.group.read_only:
            raise NgioValueError("Cannot delete a group in read only mode.")
        self.group.__delitem__("/")

    def copy_group(self, dest_group: zarr.Group):
        """Copy the group to a new store."""
        copy_group(self.group, dest_group)


def find_dimension_separator(array: zarr.Array) -> Literal[".", "/"]:
    """Find the dimension separator used in the Zarr store.

    Args:
        array (zarr.Array): The Zarr array to check.

    Returns:
        Literal[".", "/"]: The dimension separator used in the store.
    """
    from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

    if array.metadata.zarr_format == 2:
        separator = array.metadata.dimension_separator
    else:
        separator = array.metadata.chunk_key_encoding
        if not isinstance(separator, DefaultChunkKeyEncoding):
            raise NgioValueError(
                "Only DefaultChunkKeyEncoding is supported in this example."
            )
        separator = separator.separator
    return separator


def is_group_listable(group: zarr.Group) -> bool:
    """Check whether a Zarr group's contents can actually be enumerated.

    A group that was successfully opened must have its own metadata document
    (``zarr.json`` / ``.zgroup``) in the store, so a directory listing that
    does not contain it means the store cannot truly list this path (e.g. an
    HTTP host with no directory index, where zarr silently yields an empty
    listing). A genuinely empty group still lists its metadata document.

    Args:
        group (zarr.Group): The Zarr group to check.

    Returns:
        bool: True if the group's contents can be enumerated, False otherwise.
    """
    store = NgioStore.ensure(group.store)
    if not store.supports_listing:
        return False
    meta_key = "zarr.json" if group.metadata.zarr_format == 3 else ".zgroup"
    try:
        keys = store.list_dir_collected(group.path)
    except Exception:
        # Some stores may raise errors when listing
        # consider those not listable
        return False
    return meta_key in keys


def _fsspec_copy(
    src_store: NgioStore,
    src_path: str,
    dest_store: NgioStore,
    dest_path: str,
):
    src_mapper = src_store.get_mapper(src_path)
    dest_mapper = dest_store.get_mapper(dest_path)
    keys = list(src_mapper.keys())
    if not ({".zgroup", "zarr.json"} & set(keys)):
        raise NgioValueError(
            "Source listing did not return the group's metadata document; "
            "the store cannot be listed reliably, refusing to copy."
        )
    for key in keys:
        dest_mapper[key] = src_mapper[key]


def _zarr_python_copy(src_group: zarr.Group, dest_group: zarr.Group):
    # Copy attributes
    dest_group.attrs.put(src_group.attrs.asdict())
    # Copy arrays
    for name, array in src_group.arrays():
        if array.metadata.zarr_format == 2:
            spec = AnyArraySpecV2.from_zarr(array)
        else:
            spec = AnyArraySpecV3.from_zarr(array)
        dst = spec.to_zarr(
            store=dest_group.store,
            path=f"{dest_group.path}/{name}",
            overwrite=True,
        )
        if array.ndim > 0:
            if array.dtype.hasobject or array.dtype.kind in ("U", "S"):
                # dask >=2025.11 refuses auto-chunking for object/string dtypes
                # (NotImplementedError in dask.array.core.auto_chunks). These
                # arrays come from table backends and are small; copy directly.
                dst[:] = array[:]
            else:
                dask_array = da.from_zarr(array)
                da.to_zarr(dask_array, dst, overwrite=False)
    # Copy subgroups
    for name, subgroup in src_group.groups():
        dest_subgroup = dest_group.create_group(name, overwrite=True)
        _zarr_python_copy(subgroup, dest_subgroup)


def copy_group(
    src_group: zarr.Group, dest_group: zarr.Group, suppress_warnings: bool = False
):
    if src_group.metadata.zarr_format != dest_group.metadata.zarr_format:
        raise NgioValueError(
            "Different Zarr format versions between source and destination, "
            "cannot copy."
        )

    if not is_group_listable(src_group):
        raise NgioValueError("Source group is not listable, cannot copy.")

    if dest_group.read_only:
        raise NgioValueError("Destination group is read only, cannot copy.")
    src_store = NgioStore.ensure(src_group.store)
    dest_store = NgioStore.ensure(dest_group.store)
    fs_types = ("local", "fsspec")
    if src_store.store_type in fs_types and dest_store.store_type in fs_types:
        _fsspec_copy(src_store, src_group.path, dest_store, dest_group.path)
        return
    if not suppress_warnings:
        warnings.warn(
            "Fsspec copy not possible, falling back to Zarr Python API for the copy. "
            "This will preserve some tabular data non-zarr native (parquet, and csv), "
            "and it will be slower for large datasets.",
            NgioUserWarning,
            stacklevel=2,
        )
    _zarr_python_copy(src_group, dest_group)


def refresh_s3fs_config(config: NgioConfig) -> None:
    """Apply the current s3fs-related settings to s3fs's global error handler.

    Safe to call repeatedly, e.g. after mutating ``get_config().s3fs``: each call
    fully replaces s3fs's custom error handler, resetting it to a no-op when there
    are no configured retry markers instead of leaving a stale handler in place.

    A no-op when s3fs is absent, or when it predates `set_custom_error_handler`
    (added in s3fs 2026.2.0). This runs at import of `ngio.utils._zarr_utils`, and
    s3fs is not an ngio dependency, so an s3fs installed for something else must
    never break `import ngio`. The warning fires only if retry markers were
    actually configured, since otherwise there is nothing to lose.
    """
    try:
        import s3fs
    except ImportError:
        return

    custom_retry_markers = (
        config.s3fs.custom_retry_markers if config.s3fs is not None else []
    )

    if not hasattr(s3fs, "set_custom_error_handler"):
        if custom_retry_markers:
            warnings.warn(
                "s3fs.custom_retry_markers requires s3fs>=2026.2.0, but s3fs "
                f"{getattr(s3fs, '__version__', 'unknown')} is installed. The "
                "configured retry markers are ignored. Install ngio[s3] to get a "
                "supported s3fs.",
                NgioUserWarning,
                stacklevel=2,
            )
        return

    def custom_retry_handler(e: Exception) -> bool:
        """Custom s3fs retry predicate: retry errors matching a configured marker.

        Initial implementation by Paweł Rzońca @clacrow

        Receives the raw exception from botocore (before s3fs translates it, e.g.
        to PermissionError), whatever its type. Returning True makes s3fs
        sleep-and-retry the request. The motivating use case is AWS clock-skew
        ClientErrors (re-signing the request with a fresh timestamp fixes those),
        but any error substring can be configured as a retry marker.
        """
        return any(marker in str(e) for marker in custom_retry_markers)

    s3fs.set_custom_error_handler(custom_retry_handler)


refresh_s3fs_config(get_config())
