"""Common utilities for working with Zarr groups in consistent ways."""

import warnings
from pathlib import Path
from typing import Literal

import fsspec
import zarr
from filelock import BaseFileLock, FileLock
from zarr.abc.store import Store
from zarr.errors import ContainsGroupError
from zarr.storage import FsspecStore, LocalStore, MemoryStore, ZipStore

from ngio.utils._cache import NgioCache
from ngio.utils._errors import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioValueError,
)

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
# StoreLike is more restrictive than it could be
# but to make sure we can handle the store correctly
# we need to be more restrictive
NgioSupportedStore = (
    str
    | Path
    | fsspec.mapping.FSMap
    | FsspecStore
    | MemoryStore
    | LocalStore
    | ZipStore
)
GenericStore = Store | NgioSupportedStore
StoreOrGroup = GenericStore | zarr.Group


def _check_store(store) -> NgioSupportedStore:
    """Check the store and return a valid store."""
    if not isinstance(store, NgioSupportedStore):
        warnings.warn(
            f"Store type {type(store)} is not explicitly supported. "
            f"Supported types are: {NgioSupportedStore}. "
            "Proceeding, but this may lead to unexpected behavior.",
            UserWarning,
            stacklevel=2,
        )
    return store


def _check_group(
    group: zarr.Group, mode: AccessModeLiteral | None = None
) -> zarr.Group:
    """Check the group and return a valid group."""
    if group.read_only and mode not in [None, "r"]:
        raise NgioValueError(f"The group is read only. Cannot open in mode {mode}.")

    if mode == "r" and not group.read_only:
        # let's make sure we don't accidentally write to the group
        group = zarr.open_group(store=group.store, path=group.path, mode="r")
    return group


def open_group_wrapper(
    store: StoreOrGroup,
    mode: AccessModeLiteral | None = None,
    zarr_format: Literal[2, 3] | None = None,
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    Args:
        store (StoreOrGroup): The store or group to open.
        mode (AccessModeLiteral): The mode to open the group in.
        zarr_format (int): The Zarr format version to use.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        group = _check_group(store, mode)
        _check_store(group.store)
        return group

    try:
        _check_store(store)
        mode = mode if mode is not None else "a"
        group = zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)

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
            meta_mode (str): The mode of the metadata handler.
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
        self._lock: tuple[Path, BaseFileLock] | None = None

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"ZarrGroupHandler(full_url={self.full_url}, read_only={self.read_only}, "
            f"cache={self.use_cache}"
        )

    @property
    def store(self) -> Store:
        """Return the store of the group."""
        return self._group.store

    @property
    def full_url(self) -> str | None:
        """Return the store path."""
        if isinstance(self.store, LocalStore):
            return (self.store.root / self.group.path).as_posix()
        elif isinstance(self.store, FsspecStore):
            return f"{self.store.path}/{self.group.path}"
        elif isinstance(self.store, ZipStore):
            return (self.store.path / self.group.path).as_posix()
        elif isinstance(self.store, MemoryStore):
            return None
        warnings.warn(
            f"Cannot determine full URL for store type {type(self.store)}. ",
            UserWarning,
            stacklevel=2,
        )
        return None

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

        if self.use_cache is True:
            raise NgioValueError(
                "Lock mechanism is not compatible with caching. "
                "Please set cache=False to use the lock mechanism."
            )

        if not isinstance(self.store, LocalStore):
            raise NgioValueError(
                "The store needs to be a LocalStore to use the lock mechanism. "
                f"Instead, got {self.store.__class__.__name__}."
            )

        store_path = Path(self.store.root) / self.group.path
        _lock_path = store_path.with_suffix(".lock")
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

    def remove_lock(self) -> None:
        """Return the lock."""
        if self._lock is None:
            return None

        lock_path, lock = self._lock
        if lock_path.exists() and lock.lock_counter == 0:
            lock_path.unlink()
            self._lock = None
            return None

        raise NgioValueError("The lock is still in use. Cannot remove it.")

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

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        group = self.reopen_group()
        self.__init__(
            store=group,
            zarr_format=group.metadata.zarr_format,
            cache=self.use_cache,
            mode="r" if self.read_only else "r+",
        )

    @property
    def group(self) -> zarr.Group:
        """Return the group."""
        if self.use_cache is False:
            # If we are not using cache, we need to reopen the group
            # to make sure that the attributes are up to date
            return self.reopen_group()
        return self._group

    def load_attrs(self) -> dict:
        """Load the attributes of the group."""
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
        group = self.create_group(path)
        self._group_cache.set(path, group, overwrite=overwrite)
        return group

    def get_array(self, path: str) -> zarr.Array:
        """Get an array from the group."""
        array = self._array_cache.get(path)
        if isinstance(array, zarr.Array):
            return array
        array = self.group.get(path, default=None)
        if isinstance(array, zarr.Array):
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
        handler = ZarrGroupHandler(
            store=group, zarr_format=self.zarr_format, cache=self.use_cache, mode="r+"
        )
        self._handlers_cache.set(path, handler)
        return handler

    def copy_handler(self, handler: "ZarrGroupHandler") -> None:
        """Copy the group to a new store."""
        _, n_skipped, _ = zarr.copy_store(
            source=self.group.store,
            dest=handler.group.store,
            source_path=self.group.path,
            dest_path=handler.group.path,
            if_exists="replace",
        )
        if n_skipped > 0:
            raise NgioValueError(
                f"Error copying group to {handler.full_url}, "
                f"#{n_skipped} files where skipped."
            )


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
            raise ValueError(
                "Only DefaultChunkKeyEncoding is supported in this example."
            )
        separator = separator.separator
    return separator
