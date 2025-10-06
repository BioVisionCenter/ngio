"""Aggregation and filtering operations for tables."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import polars as pl

from ngio.images._ome_zarr_container import OmeZarrContainer
from ngio.tables import Table, TableType


@dataclass
class TableWithExtras:
    """A class to hold a table and its extras."""

    table: Table
    extras: dict[str, str] = field(default_factory=dict)


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

    if index_key is None:
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
) -> pl.LazyFrame:
    dataframe = dataframe.with_columns(
        [pl.lit(value, dtype=pl.String()).alias(col) for col, value in new_cols.items()]
    )

    if index_key is not None:
        dataframe = dataframe.with_columns(
            [
                pl.concat_str(
                    [pl.col(col) for col in new_cols.keys()],
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
        )
        dataframes.append(polars_ls)

    concatenated_table = pl.concat(dataframes, how="vertical")
    return concatenated_table


def conctatenate_tables(
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


def _check_images_and_extras(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
) -> None:
    """Check if the images and extras are valid."""
    if len(images) == 0:
        raise ValueError("No images to concatenate.")

    if len(images) != len(extras):
        raise ValueError("The number of images and extras must be the same.")


def _concatenate_image_tables(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType] | None = None,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
) -> Table:
    """Concatenate tables from different images into a single table."""
    _check_images_and_extras(images=images, extras=extras)

    tables = []
    for image, extra in zip(images, extras, strict=True):
        if not strict and name not in image.list_tables():
            continue
        table = image.get_table(name)
        tables.append(TableWithExtras(table=table, extras=extra))

    return conctatenate_tables(
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
    """
    return _concatenate_image_tables(
        images=images,
        extras=extras,
        name=name,
        table_cls=None,
        index_key=index_key,
        strict=strict,
        mode=mode,
    )


def concatenate_image_tables_as(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType],
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
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
    """
    table = _concatenate_image_tables(
        images=images,
        extras=extras,
        name=name,
        table_cls=table_cls,
        index_key=index_key,
        strict=strict,
        mode=mode,
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
) -> Table:
    """Concatenate tables from different images into a single table."""
    _check_images_and_extras(images=images, extras=extras)

    def process_image(
        image: OmeZarrContainer,
        name: str,
        extra: dict[str, str],
        mode: Literal["eager", "lazy"] = "eager",
        strict: bool = True,
    ) -> TableWithExtras | None:
        """Process a single image and return the table."""
        if not strict and name not in image.list_tables():
            return None
        _table = image.get_table(name)
        if mode == "lazy":
            # make sure the table is loaded lazily
            # It the backend is not lazy, this will be
            # loaded eagerly
            _ = _table.lazy_frame
        elif mode == "eager":
            # make sure the table is loaded eagerly
            _ = _table.dataframe
        table = TableWithExtras(
            table=_table,
            extras=extra,
        )
        return table

    tasks = []
    for image, extra in zip(images, extras, strict=True):
        task = asyncio.to_thread(
            process_image,
            image=image,
            name=name,
            extra=extra,
            strict=strict,
        )
        tasks.append(task)
    tables = await asyncio.gather(*tasks)
    tables = [table for table in tables if table is not None]
    return conctatenate_tables(
        tables=tables,
        mode=mode,
        index_key=index_key,
        table_cls=table_cls,
    )


async def concatenate_image_tables_async(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
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
    """
    return await _concatenate_image_tables_async(
        images=images,
        extras=extras,
        name=name,
        table_cls=None,
        index_key=index_key,
        strict=strict,
        mode=mode,
    )


async def concatenate_image_tables_as_async(
    images: Sequence[OmeZarrContainer],
    extras: Sequence[dict[str, str]],
    name: str,
    table_cls: type[TableType],
    index_key: str | None = None,
    strict: bool = True,
    mode: Literal["eager", "lazy"] = "eager",
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
    """
    table = await _concatenate_image_tables_async(
        images=images,
        extras=extras,
        name=name,
        table_cls=table_cls,
        index_key=index_key,
        strict=strict,
        mode=mode,
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
) -> list[str]:
    """List all table names in the images.

    Args:
        images: A collection of images.
        filter_types (str | None): The type of tables to filter. If None,
            return all tables. Defaults to None.
        mode (Literal["common", "all"]): Whether to return only common tables
            between all images or all tables. Defaults to "common".
    """
    tables_names = []
    for image in images:
        tables = image.list_tables(filter_types=filter_types)
        tables_names.append(tables)

    return _tables_names_coalesce(
        tables_names=tables_names,
        mode=mode,
    )


async def list_image_tables_async(
    images: Sequence[OmeZarrContainer],
    filter_types: str | None = None,
    mode: Literal["common", "all"] = "common",
) -> list[str]:
    """List all image tables in the image asynchronously.

    Args:
        images: A collection of images.
        filter_types (str | None): The type of tables to filter. If None,
            return all tables. Defaults to None.
        mode (Literal["common", "all"]): Whether to return only common tables
            between all images or all tables. Defaults to "common".
    """
    images_ids = []

    # key table name, value list of paths
    def process_image(
        image: OmeZarrContainer, filter_types: str | None = None
    ) -> list[str]:
        tables = image.list_tables(filter_types=filter_types)
        return tables

    tasks = []
    for i, image in enumerate(images):
        images_ids.append(i)
        task = asyncio.to_thread(process_image, image, filter_types=filter_types)
        tasks.append(task)

    tables_names = await asyncio.gather(*tasks)
    return _tables_names_coalesce(
        tables_names=tables_names,
        mode=mode,
    )
