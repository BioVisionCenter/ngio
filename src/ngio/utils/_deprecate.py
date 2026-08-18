"""Helpers to deprecate parts of the public ngio API."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ngio.utils._errors import NgioValueError
from ngio.utils._warnings import NgioDeprecationWarning

_F = TypeVar("_F", bound=Callable[..., Any])

DEFAULT_REMOVED_IN = "1.2"


def _qualname(func: Callable[..., Any]) -> str:
    """Return the dotted name of `func`, falling back to its repr."""
    return getattr(func, "__qualname__", repr(func))


def _resolve_aliases(
    kwargs: dict[str, Any],
    aliases: dict[str, str],
    qualname: str,
    removed_in: str,
) -> None:
    """Rewrite deprecated keywords in `kwargs` in place, warning for each."""
    for old, new in aliases.items():
        if old not in kwargs:
            continue
        if new in kwargs:
            raise NgioValueError(
                f"{qualname}() received both '{old}' and '{new}'. "
                f"'{old}' is deprecated, pass only '{new}'."
            )
        warnings.warn(
            f"The '{old}' argument of {qualname}() is deprecated and will be "
            f"removed in ngio={removed_in}. Use '{new}' instead.",
            NgioDeprecationWarning,
            stacklevel=3,
        )
        kwargs[new] = kwargs.pop(old)


def deprecated_alias(
    *, removed_in: str = DEFAULT_REMOVED_IN, **aliases: str
) -> Callable[[_F], _F]:
    """Accept renamed keyword arguments under their old spelling.

    Each `old="new"` pair forwards `old=` to `new=` and emits an
    `NgioDeprecationWarning` naming the removal version.

    Args:
        removed_in: ngio version that drops the old spelling.
        **aliases: Mapping of deprecated keyword name to its replacement.

    Raises:
        NgioValueError: If both spellings are passed in the same call.

    Note:
        On an `async def` callable the warning fires when the coroutine is
        created, not when it is awaited.

    Example:
        ```python
        @deprecated_alias(levels_paths="level_paths")
        def from_shapes(level_paths: Sequence[str] | None = None): ...
        ```
    """

    def decorator(func: _F) -> _F:
        qualname = _qualname(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _resolve_aliases(kwargs, aliases, qualname, removed_in)
            return func(*args, **kwargs)

        return cast("_F", wrapper)

    return decorator


def deprecated(
    *, replacement: str, removed_in: str = DEFAULT_REMOVED_IN
) -> Callable[[_F], _F]:
    """Mark a callable as deprecated, pointing callers at its replacement.

    Args:
        replacement: What to use instead, quoted verbatim in the warning.
        removed_in: ngio version that drops the callable.

    Note:
        On an `async def` callable the warning fires when the coroutine is
        created, not when it is awaited, so the warning always carries the
        caller's stack frame.

    Example:
        ```python
        @deprecated(replacement="get_images(max_workers=...)")
        async def get_images_async(self): ...
        ```
    """

    def decorator(func: _F) -> _F:
        message = (
            f"{_qualname(func)}() is deprecated and will be removed in "
            f"ngio={removed_in}. Use {replacement} instead."
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, NgioDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return cast("_F", wrapper)

    return decorator
