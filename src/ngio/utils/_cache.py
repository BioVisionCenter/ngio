from typing import Generic, TypeVar

T = TypeVar("T")


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

    def set(self, key: str, value: T, overwrite: bool = True) -> None:
        if not self._use_cache:
            self._cache_sanity_check()
            return
        self._cache[key] = value

    def clear(self) -> None:
        if not self._use_cache:
            self._cache_sanity_check()
            return
        self._cache.clear()
