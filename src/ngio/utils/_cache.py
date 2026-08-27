import warnings
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_UNSET: Any = object()


class NgioCache(Generic[T]):
    """A simple cache for NGIO objects."""

    def __init__(self, use_cache: bool = True):
        self._cache: dict[str, T] = {}
        self._use_cache = use_cache

    def _cache_sanity_check(self) -> None:
        if len(self._cache) > 0:
            raise RuntimeError(
                "Cache is disabled, but cache contains items. "
                "This indicates a logic error."
            )

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def cache(self) -> dict[str, T]:
        return self._cache

    @property
    def is_empty(self) -> bool:
        return len(self._cache) == 0

    def get(self, key: str, default: T | None = None) -> T | None:
        if not self._use_cache:
            self._cache_sanity_check()
            return default
        return self._cache.get(key, default)

    def set(self, key: str, value: T, overwrite: bool = _UNSET) -> None:
        if overwrite is not _UNSET:
            # Shipped in 1.0 and always ignored; kept as a warning no-op
            # through the 1.1 cycle rather than removed outright.
            from ngio.utils._warnings import NgioDeprecationWarning

            warnings.warn(
                "The 'overwrite' argument of NgioCache.set() is deprecated "
                "and will be removed in ngio=1.2. It was always ignored: "
                "'set' always overwrites.",
                NgioDeprecationWarning,
                stacklevel=2,
            )
        if not self._use_cache:
            self._cache_sanity_check()
            return
        self._cache[key] = value

    def setdefault(self, key: str, value: T) -> T:
        """Insert `value` if the key is absent, and return the cached winner.

        A single atomic dict operation, so concurrent builders of the same
        key end up sharing one object instead of each keeping their own —
        which is what keeps "repeated gets return the identical object" true
        under a threaded fan-out.
        """
        if not self._use_cache:
            self._cache_sanity_check()
            return value
        return self._cache.setdefault(key, value)

    def pop(self, key: str) -> T | None:
        """Drop one entry, returning it (or `None` when absent)."""
        if not self._use_cache:
            self._cache_sanity_check()
            return None
        return self._cache.pop(key, None)

    def clear(self) -> None:
        if not self._use_cache:
            self._cache_sanity_check()
            return
        self._cache.clear()
