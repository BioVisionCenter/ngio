"""Common utilities for working with Zarr groups in consistent ways."""

from pathlib import Path
from typing import Literal

import fsspec
import zarr
from filelock import BaseFileLock, FileLock
from zarr.abc.store import Store
from zarr.core.array import CompressorLike
from zarr.errors import ContainsGroupError
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from ngio.utils import NgioFileExistsError, NgioFileNotFoundError, NgioValueError
from ngio.utils._errors import NgioError

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
# StoreLike is more restrictive than it could be
# but to make sure we can handle the store correctly
# we need to be more restrictive
NgioSupportedStore = (
    str | Path | fsspec.mapping.FSMap | FsspecStore | MemoryStore | LocalStore
)
GenericStore = Store | NgioSupportedStore
StoreOrGroup = GenericStore | zarr.Group


def _check_store(store) -> NgioSupportedStore:
    """Check the store and return a valid store."""
    if isinstance(store, NgioSupportedStore):
        return store

    raise NotImplementedError(
        f"Store type {type(store)} is not supported. "
        f"Supported types are: {NgioSupportedStore}"
    )


def _check_group(group: zarr.Group, mode: AccessModeLiteral) -> zarr.Group:
    """Check the group and return a valid group."""
    if group.read_only and mode in ["w", "w-"]:
        raise NgioValueError(
            "The group is read only. Cannot open in write mode ['w', 'w-']"
        )

    if mode == "r" and not group.read_only:
        # let's make sure we don't accidentally write to the group
        group = zarr.open_group(store=group.store, path=group.path, mode="r")

    return group


def open_group_wrapper(
    store: StoreOrGroup,
    mode: AccessModeLiteral,
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
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
        parent: "ZarrGroupHandler | None" = None,
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            zarr_format (int): The Zarr format version to use.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
            parallel_safe (bool): If True, the handler will create a lock file to make
                that can be used to make the handler parallel safe.
                Be aware that the lock needs to be used manually.
            parent (ZarrGroupHandler | None): The parent handler.
        """
        if mode not in ["r", "r+", "w", "w-", "a"]:
            raise NgioValueError(f"Mode {mode} is not supported.")

        if parallel_safe and cache:
            raise NgioValueError(
                "The cache and parallel_safe options are mutually exclusive."
                "If you want to use the lock mechanism, you should not use the cache."
            )

        group = open_group_wrapper(store=store, mode=mode, zarr_format=zarr_format)
        _store = group.store

        # Make sure the cache is set in the attrs
        # in the same way as the cache in the handler

        ## TODO
        # Figure out how to handle the cache in the new zarr version
        # group.attrs.cache = cache

        if parallel_safe:
            if not isinstance(_store, LocalStore):
                raise NgioValueError(
                    "The store needs to be a LocalStore to use the lock mechanism. "
                    f"Instead, got {_store.__class__.__name__}."
                )

            store_path = _store.root / group.path
            self._lock_path = store_path.with_suffix(".lock")
            self._lock = FileLock(self._lock_path, timeout=10)

        else:
            self._lock_path = None
            self._lock = None

        self._group = group
        self._mode = mode
        self.use_cache = cache
        self._parallel_safe = parallel_safe
        self._cache = {}
        self._parent = parent

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"ZarrGroupHandler(full_url={self.full_url}, mode={self.mode}, "
            f"cache={self.use_cache}"
        )

    @property
    def store(self) -> Store:
        """Return the store of the group."""
        return self.group.store

    @property
    def full_url(self) -> str | None:
        """Return the store path."""
        if isinstance(self.store, LocalStore):
            return (self.store.root / self.group.path).as_posix()
        if isinstance(self.store, FsspecStore):
            return self.store.fs.map.root_path
        return None

    @property
    def zarr_format(self) -> Literal[2, 3]:
        """Return the Zarr format version."""
        return self.group.metadata.zarr_format

    @property
    def mode(self) -> AccessModeLiteral:
        """Return the mode of the group."""
        return self._mode  # type: ignore

    @property
    def lock(self) -> BaseFileLock:
        """Return the lock."""
        if self._lock is None:
            raise NgioValueError(
                "The handler is not parallel safe. "
                "Reopen the handler with parallel_safe=True."
            )
        return self._lock

    @property
    def parent(self) -> "ZarrGroupHandler | None":
        """Return the parent handler."""
        return self._parent

    def remove_lock(self) -> None:
        """Return the lock."""
        if self._lock is None or self._lock_path is None:
            return None

        lock_path = Path(self._lock_path)
        if lock_path.exists() and self._lock.lock_counter == 0:
            lock_path.unlink()
            self._lock = None
            self._lock_path = None
            return None

        raise NgioValueError("The lock is still in use. Cannot remove it.")

    @property
    def group(self) -> zarr.Group:
        """Return the group."""
        if self._parallel_safe:
            # If we are parallel safe, we need to reopen the group
            # to make sure that the attributes are up to date
            if self.mode == "r":
                mode = "r"
            else:
                mode = "r+"
            return zarr.open_group(
                store=self._group.store,
                path=self._group.path,
                mode=mode,
                zarr_format=self._group.metadata.zarr_format,
            )
        return self._group

    def add_to_cache(self, key: str, value: object) -> None:
        """Add an object to the cache."""
        if not self.use_cache:
            return None
        self._cache[key] = value

    def get_from_cache(self, key: str) -> object | None:
        """Get an object from the cache."""
        if not self.use_cache:
            return None
        return self._cache.get(key, None)

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        self._cache = {}

    def load_attrs(self) -> dict:
        """Load the attributes of the group."""
        attrs = self.get_from_cache("attrs")
        if attrs is not None and isinstance(attrs, dict):
            return attrs

        attrs = dict(self.group.attrs)

        self.add_to_cache("attrs", attrs)
        return attrs

    def _write_attrs(self, attrs: dict, overwrite: bool = False) -> None:
        """Write the metadata to the store."""
        if self.group.read_only:
            raise NgioValueError("The group is read only. Cannot write metadata.")

        # we need to invalidate the current attrs cache
        self.add_to_cache("attrs", None)
        if overwrite:
            self.group.attrs.clear()

        self.group.attrs.update(attrs)

    def write_attrs(self, attrs: dict, overwrite: bool = False) -> None:
        """Write the metadata to the store."""
        # Maybe we should use the lock here
        self._write_attrs(attrs, overwrite)

    def _obj_get(self, path: str):
        """Get a group from the group."""
        group_or_array = self.get_from_cache(path)
        if group_or_array is not None:
            return group_or_array

        group_or_array = self.group.get(path, None)
        self.add_to_cache(path, group_or_array)
        return group_or_array

    def create_group(self, path: str, overwrite: bool = False) -> zarr.Group:
        """Create a group in the group."""
        if self.mode == "r":
            raise NgioValueError("Cannot create a group in read only mode.")

        try:
            group = self.group.create_group(path, overwrite=overwrite)
        except ContainsGroupError as e:
            raise NgioFileExistsError(
                f"A Zarr group already exists at {path}, "
                "consider setting overwrite=True."
            ) from e
        self.add_to_cache(path, group)
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

        group = self._obj_get(path)
        if isinstance(group, zarr.Group):
            return group

        if group is not None:
            raise NgioValueError(
                f"The object at {path} is not a group, but a {type(group)}"
            )

        if not create_mode:
            raise NgioFileNotFoundError(f"No group found at {path}")
        group = self.create_group(path)
        return group

    def safe_get_group(
        self, path: str, create_mode: bool = False
    ) -> tuple[bool, zarr.Group | NgioError]:
        """Get a group from the group.

        Args:
            path (str): The path to the group.
            create_mode (bool): If True, create the group if it does not exist.

        Returns:
            zarr.Group | None: The Zarr group or None if it does not exist
                or an error occurs.

        """
        try:
            return True, self.get_group(path, create_mode)
        except NgioError as e:
            return False, e

    def get_array(self, path: str) -> zarr.Array:
        """Get an array from the group."""
        array = self._obj_get(path)
        if array is None:
            raise NgioFileNotFoundError(f"No array found at {path}")
        if not isinstance(array, zarr.Array):
            raise NgioValueError(
                f"The object at {path} is not an array, but a {type(array)}"
            )
        return array

    def create_array(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: str,
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        compressors: CompressorLike = "auto",
        separator: Literal[".", "/"] = "/",
        overwrite: bool = False,
    ) -> zarr.Array:
        if self.mode == "r":
            raise NgioValueError("Cannot create an array in read only mode.")

        if self.zarr_format == 2:
            chunks_encoding = {
                "name": "v2",
                "separator": separator,
            }
        else:
            chunks_encoding = {
                "name": "default",
                "separator": separator,
            }

        try:
            return self.group.create_array(
                name=path,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                chunk_key_encoding=chunks_encoding,
                overwrite=overwrite,
                compressors=compressors,
            )
        except ContainsGroupError as e:
            raise NgioFileExistsError(
                f"A Zarr array already exists at {path}, "
                "consider setting overwrite=True."
            ) from e
        except Exception as e:
            raise NgioValueError(f"Error creating array at {path}") from e

    def derive_handler(
        self,
        path: str,
        overwrite: bool = False,
    ) -> "ZarrGroupHandler":
        """Derive a new handler from the current handler.

        Args:
            path (str): The path to the group.
            overwrite (bool): If True, overwrite the group if it exists.
        """
        group = self.get_group(path, create_mode=True, overwrite=overwrite)
        return ZarrGroupHandler(
            store=group,
            zarr_format=self.zarr_format,
            cache=self.use_cache,
            mode=self.mode,
            parallel_safe=self._parallel_safe,
            parent=self,
        )

    def safe_derive_handler(
        self,
        path: str,
        overwrite: bool = False,
    ) -> tuple[bool, "ZarrGroupHandler | NgioError"]:
        """Derive a new handler from the current handler."""
        try:
            return True, self.derive_handler(path, overwrite=overwrite)
        except NgioError as e:
            return False, e

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
