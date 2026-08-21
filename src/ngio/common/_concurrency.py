"""Shared `max_workers` machinery for ngio's thread fan-outs.

Hoisted from `images/_table_ops.py`, where it grew up around the plate-wide
table operations: the same pool sizing and the same future-default warning now
also serve the parallel iterator mappers, and a pixel path importing from a
table module would be inverted.
"""

import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypeVar

from ngio.utils import NgioFutureWarning, NgioValueError
from ngio.utils._warnings import stacklevel_of_first_caller

_T = TypeVar("_T")
_R = TypeVar("_R")

#: Accepted by every `max_workers` argument. `None` means "unspecified" and
#: currently runs serially; `1` is serial deliberately; `"auto"` picks a pool
#: sized for round-trip-bound work.
MaxWorkers = int | Literal["auto"] | None

#: The release in which `max_workers` starts defaulting to `"auto"`.
_DEFAULT_CHANGES_IN = "1.2"


def _warn_default_will_change(n_items: int) -> None:
    """Announce the coming default, only where it will actually matter.

    Deduplication is left to the `warnings` module, which suppresses repeats
    per call site. A hand-rolled once-per-process flag would be worse: under
    `filterwarnings = ["error"]` only the first caller in the process raises,
    so which test fails depends on collection order.
    """
    if n_items <= 1:
        return
    warnings.warn(
        "Plate-wide operations still read one item at a time by default. In "
        f"ngio={_DEFAULT_CHANGES_IN} the default for `max_workers` changes from "
        '`None` to `"auto"`, which reads them concurrently -- several times '
        "faster on a remote store, where these calls are round-trip bound. "
        'Pass `max_workers="auto"` to opt in now, or `max_workers=1` to keep '
        "reading serially and silence this.",
        NgioFutureWarning,
        stacklevel=stacklevel_of_first_caller(),
    )


def _resolve_max_workers(max_workers: MaxWorkers) -> int | None:
    """Turn `"auto"` into a concrete pool size.

    These fan-outs wait on store round-trips rather than on the CPU, so the
    useful pool is far wider than the core count. This is the same cap asyncio
    puts on its default executor, which `_gather_bounded` already inherits.
    """
    if max_workers == "auto":
        return min(32, (os.cpu_count() or 1) + 4)
    return max_workers


def _map_workers(
    func: Callable[[_T], _R],
    items: Sequence[_T],
    max_workers: MaxWorkers,
) -> list[_R]:
    """Apply `func` to every item, on a thread pool when `max_workers` > 1.

    Results keep the order of `items`. With `max_workers=None` (the default)
    the work runs serially in the calling thread; `"auto"` sizes the pool for
    round-trip-bound work, and `1` is serial by explicit request.
    """
    if max_workers is None:
        _warn_default_will_change(len(items))
    elif isinstance(max_workers, int) and max_workers < 1:
        raise NgioValueError(
            f"max_workers must be >= 1, got {max_workers}. Use 1 for serial "
            'execution, or "auto" to size the pool automatically.'
        )
    max_workers = _resolve_max_workers(max_workers)
    if max_workers is None or max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(func, items))
