"""Aggregation and filtering operations for tables."""

import asyncio
import os
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from inspect import currentframe
from pathlib import Path
from typing import Literal, TypeVar

import pandas as pd
import polars as pl

from ngio.images._ome_zarr_container import OmeZarrContainer
from ngio.tables import Table, TableType
from ngio.utils import NgioFutureWarning, deprecated

_T = TypeVar("_T")
_R = TypeVar("_R")

#: Accepted by every `max_workers` argument. `None` means "unspecified" and
#: currently runs serially; `1` is serial deliberately; `"auto"` picks a pool
#: sized for round-trip-bound work.
MaxWorkers = int | Literal["auto"] | None

#: The release in which `max_workers` starts defaulting to `"auto"`.
_DEFAULT_CHANGES_IN = "1.2"


def _stacklevel_of_first_caller() -> int:
    """Return the `stacklevel` that points at the first frame outside ngio.

    A fixed level cannot work: `get_wells()` reaches the fan-out through two
    ngio frames and `images_paths()` through three, so any constant blames
    ngio's own source for one of them. That matters twice over — the reader is
    told to edit a file they do not own, and the `warnings` module keys its
    deduplication on the reported location, so every caller would collapse onto
    one entry.
    """
    package_root = str(Path(__file__).parents[1])
    frame = currentframe()
    level = 0
    while frame is not None:
        level += 1
        frame = frame.f_back
        if frame is not None and not frame.f_code.co_filename.startswith(package_root):
            return level
    return 2


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
        stacklevel=_stacklevel_of_first_caller(),
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


@dataclass
class TableWithExtras:
    """A class to hold a table and its extras."""

    table: Table
    extras: dict[str, str] = field(default_factory=dict)


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
    max_workers = _resolve_max_workers(max_workers)
    if max_workers is None or max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(func, items))


async def _gather_bounded(
    coro_factories: Sequence[Callable[[], Awaitable[_R]]],
    max_workers: MaxWorkers,
) -> list[_R]:
    """Await every coroutine, at most `max_workers` of them at a time.

    With `max_workers=None` the fan-out is left to asyncio's default thread
    executor, which is itself capped at `min(32, cpu_count + 4)`.
    """
    if max_workers is None:
        _warn_default_will_change(len(coro_factories))
    max_workers = _resolve_max_workers(max_workers)
    if max_workers is None or max_workers <= 0:
        return list(await asyncio.gather(*(factory() for factory in coro_factories)))

    semaphore = asyncio.Semaphore(max_workers)

    async def _run(factory: Callable[[], Awaitable[_R]]) -> _R:
        async with semaphore:
            return await factory()

    return list(await asyncio.gather(*(_run(f) for f in coro_factories)))


def _reindex_dataframe(
    dataframe, index_cols: list[str], index_key: str | None = None
) -> pd.DataFrame:
    """Reindex a dataframe using an hash of the index columns."""
    # Reindex the dataframe
    old_index = dataframe.index.name
    if old_index is not None:
        dataframe = dataframe.reset_index()
        index_cols.append(old_index)
    dataframe.index = dataframe[index_cols].astype(str).agg("_".join, axis=1)

    if index_key is not None:
        dataframe.index.name = index_key
    return dataframe


def _add_const_columns(
    dataframe: pd.DataFrame,
    new_cols: dict[str, str],
    index_key: str | None = None,
) -> pd.DataFrame:
    for col, value in new_cols.items():
        dataframe[col] = value

    if index_key is not None:
        dataframe = _reindex_dataframe(
            dataframe=dataframe,
            index_cols=list(new_cols.keys()),
            index_key=index_key,
        )
    return dataframe


def _add_const_columns_pl(
    dataframe: pl.LazyFrame,
    new_cols: dict[str, str],
    index_key: str | None = None,
    table_index_key: str | None = None,
) -> pl.LazyFrame:
    dataframe = dataframe.with_columns(
        [pl.lit(value, dtype=pl.String()).alias(col) for col, value in new_cols.items()]
    )

    if index_key is not None:
        # Mirror the pandas path: the new index hashes the extras columns
        # plus the table's original index column (when present).
        index_cols = list(new_cols.keys())
        if (
            table_index_key is not None
            and table_index_key in dataframe.collect_schema().names()
        ):
            index_cols.append(table_index_key)
        dataframe = dataframe.with_columns(
            [
                pl.concat_str(
                    [pl.col(col).cast(pl.String()) for col in index_cols],
                    separator="_",
                ).alias(index_key)
            ]
        )
    return dataframe


def _pd_concat(
    tables: Sequence[TableWithExtras], index_key: str | None = None
) -> pd.DataFrame:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    dataframes = []
    for table in tables:
        dataframe = _add_const_columns(
            dataframe=table.table.dataframe, new_cols=table.extras, index_key=index_key
        )
        dataframes.append(dataframe)
    concatenated_table = pd.concat(dataframes, axis=0)
    return concatenated_table


def _pl_concat(
    tables: Sequence[TableWithExtras], index_key: str | None = None
) -> pl.LazyFrame:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    dataframes = []
    for table in tables:
        polars_ls = _add_const_columns_pl(
            dataframe=table.table.lazy_frame,
            new_cols=table.extras,
            index_key=index_key,
            table_index_key=table.table.meta.index_key,
        )
        dataframes.append(polars_ls)

    concatenated_table = pl.concat(dataframes, how="vertical")
    return concatenated_table


def concatenate_tables(
    tables: Sequence[TableWithExtras],
    mode: Literal["eager", "lazy"] = "eager",
    index_key: str | None = None,
    table_cls: type[TableType] | None = None,
) -> Table:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    table0 = next(iter(tables)).table

    if mode == "lazy":
        concatenated_table = _pl_concat(tables=tables, index_key=index_key)
    elif mode == "eager":
        concatenated_table = _pd_concat(tables=tables, index_key=index_key)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'eager' or 'lazy'.")

    meta = table0.meta
    meta.index_key = index_key
    meta.index_type = "str"

    if table_cls is not None:
        return table_cls.from_table_data(
            table_data=concatenated_table,
            meta=meta,
        )
    return table0.from_table_data(
        table_data=concatenated_table,
        meta=meta,
    )


conctatenate_tables = deprecated(replacement="concatenate_tables()")(concatenate_tables)
"""Deprecated misspelling of `concatenate_tables`."""


def _check_images_and_extras(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
) -> None:
    """Check if the images and extras are valid."""
    if len(images) == 0:
        raise ValueError("No images to concatenate.")

    if len(images) != len(extras):
        raise ValueError("The number of images and extras must be the same.")


def _load_image_table(
    image: OmeZarrContainer,
    name: str,
    extra: dict[str, str],
    mode: Literal["eager", "lazy"] = "eager",
    strict: bool = True,
) -> TableWithExtras | None:
    """Load `name` from `image` and materialize it according to `mode`.

    The materialization is what makes this worth running on a worker thread:
    it forces the IO here instead of leaving it to the concatenation step.
    """
    if not strict and name not in image.list_tables():
        return None
    table = image.get_table(name)
    if mode == "lazy":
        # If the backend has no lazy path this loads eagerly.
        _ = table.lazy_frame
    elif mode == "eager":
        _ = table.dataframe
    return TableWithExtras(table=table, extras=extra)


def _concatenate_image_tables(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType] | None = None,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> Table:
    """Concatenate tables from different images into a single table."""
    _check_images_and_extras(images=images, extras=extras)

    pairs = list(zip(images, extras, strict=True))
    loaded = _map_workers(
        lambda pair: _load_image_table(
            image=pair[0], name=name, extra=pair[1], mode=mode, strict=strict
        ),
        pairs,
        max_workers=max_workers,
    )
    tables = [table for table in loaded if table is not None]

    return concatenate_tables(
        tables=tables,
        mode=mode,
        index_key=index_key,
        table_cls=table_cls,
    )


def concatenate_image_tables(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> Table:
    """Concatenate tables from different images into a single table.

    Args:
        images: A collection of images.
        extras: A collection of extras dictionaries for each image.
            this will be added as columns to the table, and will be
            concatenated with the table index to create a new index.
        name: The name of the table to concatenate.
        index_key: The key to use for the index of the concatenated table.
        strict: If True, raise an error if the table is not found in the image.
        mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
            if 'eager', the table will be loaded into memory.
            if 'lazy', the table will be loaded as a lazy frame.
        max_workers: How many images to read concurrently. `None` (the default)
            reads them one at a time in the calling thread.
    """
    return _concatenate_image_tables(
        images=images,
        extras=extras,
        name=name,
        table_cls=None,
        index_key=index_key,
        strict=strict,
        mode=mode,
        max_workers=max_workers,
    )


def concatenate_image_tables_as(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType],
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> TableType:
    """Concatenate tables from different images into a single table.

    Args:
        images: A collection of images.
        extras: A collection of extras dictionaries for each image.
            this will be added as columns to the table, and will be
            concatenated with the table index to create a new index.
        name: The name of the table to concatenate.
        table_cls: The output will be casted to this class, if the new table_cls is
            compatible with the table_cls of the input tables.
        index_key: The key to use for the index of the concatenated table.
        strict: If True, raise an error if the table is not found in the image.
        mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
            if 'eager', the table will be loaded into memory.
            if 'lazy', the table will be loaded as a lazy frame.
        max_workers: How many images to read concurrently. `None` (the default)
            reads them one at a time in the calling thread.
    """
    table = _concatenate_image_tables(
        images=images,
        extras=extras,
        name=name,
        table_cls=table_cls,
        index_key=index_key,
        strict=strict,
        mode=mode,
        max_workers=max_workers,
    )
    if not isinstance(table, table_cls):
        raise ValueError(f"Table is not of type {table_cls}. Got {type(table)}")
    return table


async def _concatenate_image_tables_async(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType] | None = None,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> Table:
    """Concatenate tables from different images into a single table."""
    _check_images_and_extras(images=images, extras=extras)

    factories = [
        (
            lambda image=image, extra=extra: asyncio.to_thread(
                _load_image_table,
                image=image,
                name=name,
                extra=extra,
                mode=mode,
                strict=strict,
            )
        )
        for image, extra in zip(images, extras, strict=True)
    ]
    loaded = await _gather_bounded(factories, max_workers=max_workers)
    tables = [table for table in loaded if table is not None]
    return concatenate_tables(
        tables=tables,
        mode=mode,
        index_key=index_key,
        table_cls=table_cls,
    )


@deprecated(replacement="concatenate_image_tables(max_workers=...)")
async def concatenate_image_tables_async(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> Table:
    """Concatenate tables from different images into a single table.

    Deprecated: use `concatenate_image_tables(max_workers=...)`.

    Args:
        images: A collection of images.
        extras: A collection of extras dictionaries for each image.
            this will be added as columns to the table, and will be
            concatenated with the table index to create a new index.
        name: The name of the table to concatenate.
        index_key: The key to use for the index of the concatenated table.
        strict: If True, raise an error if the table is not found in the image.
        mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
            if 'eager', the table will be loaded into memory.
            if 'lazy', the table will be loaded as a lazy frame.
        max_workers: How many images to read at a time. `None` leaves the
            fan-out to asyncio's default thread executor.
    """
    return await _concatenate_image_tables_async(
        images=images,
        extras=extras,
        name=name,
        table_cls=None,
        index_key=index_key,
        strict=strict,
        mode=mode,
        max_workers=max_workers,
    )


@deprecated(replacement="concatenate_image_tables_as(max_workers=...)")
async def concatenate_image_tables_as_async(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType],
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
    max_workers: MaxWorkers = None,
) -> TableType:
    """Concatenate tables from different images into a single table.

    Deprecated: use `concatenate_image_tables_as(max_workers=...)`.

    Args:
        images: A collection of images.
        extras: A collection of extras dictionaries for each image.
            this will be added as columns to the table, and will be
            concatenated with the table index to create a new index.
        name: The name of the table to concatenate.
        table_cls: The output will be casted to this class, if the new table_cls is
            compatible with the table_cls of the input tables.
        index_key: The key to use for the index of the concatenated table.
        strict: If True, raise an error if the table is not found in the image.
        mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
            if 'eager', the table will be loaded into memory.
            if 'lazy', the table will be loaded as a lazy frame.
        max_workers: How many images to read at a time. `None` leaves the
            fan-out to asyncio's default thread executor.
    """
    table = await _concatenate_image_tables_async(
        images=images,
        extras=extras,
        name=name,
        table_cls=table_cls,
        index_key=index_key,
        strict=strict,
        mode=mode,
        max_workers=max_workers,
    )
    if not isinstance(table, table_cls):
        raise ValueError(f"Table is not of type {table_cls}. Got {type(table)}")
    return table


def _tables_names_coalesce(
    tables_names: list[list[str]],
    mode: Literal["common", "all"] = "common",
) -> list[str]:
    num_images = len(tables_names)
    if num_images == 0:
        raise ValueError("No images to concatenate.")

    names = [name for _names in tables_names for name in _names]
    names_counts = Counter(names)

    if mode == "common":
        # Get the names that are present in all images
        common_names = [
            name for name, count in names_counts.items() if count == num_images
        ]
        return common_names
    elif mode == "all":
        # Get all names
        return list(names_counts.keys())
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'common' or 'all'.")


def list_image_tables(
    images: Sequence[OmeZarrContainer],
    filter_types: str | None = None,
    mode: Literal["common", "all"] = "common",
    max_workers: MaxWorkers = None,
) -> list[str]:
    """List all table names in the images.

    Args:
        images: A collection of images.
        filter_types: The type of tables to filter. If None, return all tables.
        mode: Whether to return only tables common to every image (`"common"`)
            or the union across them (`"all"`).
        max_workers: How many images to read concurrently. `None` (the default)
            reads them one at a time in the calling thread.
    """
    tables_names = _map_workers(
        lambda image: image.list_tables(filter_types=filter_types),
        list(images),
        max_workers=max_workers,
    )
    return _tables_names_coalesce(
        tables_names=tables_names,
        mode=mode,
    )


async def _list_image_tables_async(
    images: Sequence[OmeZarrContainer],
    filter_types: str | None = None,
    mode: Literal["common", "all"] = "common",
    max_workers: MaxWorkers = None,
) -> list[str]:
    """List all table names in the images, reading them off the event loop."""
    factories = [
        (
            lambda image=image: asyncio.to_thread(
                image.list_tables, filter_types=filter_types
            )
        )
        for image in images
    ]
    tables_names = await _gather_bounded(factories, max_workers=max_workers)
    return _tables_names_coalesce(
        tables_names=tables_names,
        mode=mode,
    )


@deprecated(replacement="list_image_tables(max_workers=...)")
async def list_image_tables_async(
    images: Sequence[OmeZarrContainer],
    filter_types: str | None = None,
    mode: Literal["common", "all"] = "common",
    max_workers: MaxWorkers = None,
) -> list[str]:
    """List all image tables in the images asynchronously.

    Deprecated: use `list_image_tables(max_workers=...)`.

    Args:
        images: A collection of images.
        filter_types: The type of tables to filter. If None, return all tables.
        mode: Whether to return only tables common to every image (`"common"`)
            or the union across them (`"all"`).
        max_workers: How many images to read concurrently. `None` (the default)
            leaves the fan-out to asyncio's default thread executor.
    """
    return await _list_image_tables_async(
        images=images,
        filter_types=filter_types,
        mode=mode,
        max_workers=max_workers,
    )
