"""Configurable retry helpers for ngio IO operations.

The retry policy is defined by `ngio.config.RetryConfig`. With the default
policy (`max_retries=0`) every helper here degrades to a plain call.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from ngio.config import RetryConfig, get_config
from ngio.utils._errors import NgioError

logger = logging.getLogger(f"ngio:{__name__}")

T = TypeVar("T")
P = ParamSpec("P")

# Kept as a module attribute so tests can monkeypatch it.
_sleep = time.sleep

_NEVER_RETRY: tuple[type[BaseException], ...] = (
    NgioError,
    KeyboardInterrupt,
    SystemExit,
    asyncio.CancelledError,
    GeneratorExit,
)


def is_retryable(exc: BaseException, policy: RetryConfig) -> bool:
    """Return whether the error should be retried under the given policy.

    Ngio's own errors and interpreter-control exceptions are never retried,
    even when `retry_all_errors` is enabled.
    """
    if not isinstance(exc, Exception) or isinstance(exc, _NEVER_RETRY):
        return False
    if policy.retry_all_errors:
        return True
    error_repr = f"{type(exc).__name__}: {exc}"
    return any(marker in error_repr for marker in policy.retry_on)


def _log_retry(
    op_name: str, attempt: int, policy: RetryConfig, delay: float, exc: Exception
) -> None:
    logger.warning(
        "Retrying %s after error '%s: %s' (attempt %d/%d, sleeping %.2fs)",
        op_name or "IO operation",
        type(exc).__name__,
        exc,
        attempt,
        policy.max_retries,
        delay,
    )


async def aretry_call(
    fn: Callable[[], Awaitable[T]], policy: RetryConfig, *, op_name: str = ""
) -> T:
    """Await `fn()` retrying retryable errors according to the policy.

    Note:
        Never use blocking sleeps here: this runs on zarr's single IO event
        loop, where `asyncio.sleep` suspends only the retrying coroutine.
    """
    for attempt in range(1, policy.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            if not is_retryable(e, policy):
                raise
            delay = policy.backoff.compute_delay(attempt)
            _log_retry(op_name, attempt, policy, delay, e)
            await asyncio.sleep(delay)
    return await fn()


def retry_call(fn: Callable[[], T], policy: RetryConfig, *, op_name: str = "") -> T:
    """Call `fn()` retrying retryable errors according to the policy."""
    for attempt in range(1, policy.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e, policy):
                raise
            delay = policy.backoff.compute_delay(attempt)
            _log_retry(op_name, attempt, policy, delay, e)
            _sleep(delay)
    return fn()


def retry_io(fn: Callable[P, T]) -> Callable[P, T]:
    """Retry a synchronous IO function according to the global io_retry config.

    The policy is read from `get_config().io_retry` at every call, so runtime
    changes to the global config take effect immediately.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        policy = get_config().io_retry
        if policy.max_retries == 0:
            return fn(*args, **kwargs)
        op_name = getattr(fn, "__qualname__", repr(fn))
        return retry_call(lambda: fn(*args, **kwargs), policy, op_name=op_name)

    return wrapper
