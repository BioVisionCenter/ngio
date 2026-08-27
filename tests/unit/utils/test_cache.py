import pytest

from ngio.utils import NgioCache, NgioDeprecationWarning


def test_cache_enabled():
    cache: NgioCache[int] = NgioCache(use_cache=True)
    assert cache.use_cache
    assert cache.is_empty
    assert cache.cache == {}

    cache.set("a", 1)
    assert not cache.is_empty
    assert cache.get("a") == 1
    assert cache.cache == {"a": 1}
    assert cache.get("missing") is None
    assert cache.get("missing", default=42) == 42

    cache.set("a", 2)
    assert cache.get("a") == 2

    cache.clear()
    assert cache.is_empty
    assert cache.get("a") is None


def test_cache_pop():
    cache: NgioCache[int] = NgioCache(use_cache=True)
    cache.set("a", 1)
    assert cache.pop("a") == 1
    assert cache.pop("a") is None
    assert cache.is_empty

    disabled: NgioCache[int] = NgioCache(use_cache=False)
    assert disabled.pop("a") is None


def test_cache_set_overwrite_kwarg_is_a_deprecated_no_op():
    cache: NgioCache[int] = NgioCache(use_cache=True)
    cache.set("a", 1)
    with pytest.warns(NgioDeprecationWarning, match="always ignored"):
        cache.set("a", 2, overwrite=False)
    # Unchanged behaviour: `set` always overwrites, as it did in 1.0.
    assert cache.get("a") == 2


def test_cache_disabled():
    cache: NgioCache[int] = NgioCache(use_cache=False)
    assert not cache.use_cache

    # All operations are no-ops when the cache is disabled
    cache.set("a", 1)
    assert cache.is_empty
    assert cache.get("a") is None
    assert cache.get("a", default=7) == 7

    cache.clear()
    assert cache.is_empty


def test_cache_disabled_sanity_check():
    cache: NgioCache[int] = NgioCache(use_cache=False)
    # Simulate a logic error: items in the internal dict while disabled
    cache._cache["a"] = 1

    with pytest.raises(RuntimeError, match="Cache is disabled"):
        cache.get("a")
    with pytest.raises(RuntimeError, match="Cache is disabled"):
        cache.set("b", 2)
    with pytest.raises(RuntimeError, match="Cache is disabled"):
        cache.clear()
