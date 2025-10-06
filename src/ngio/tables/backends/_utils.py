"""Utility functions for converting between different tables formats.

The supported formats are:
- pandas DataFrame
- polars DataFrame or LazyFrame
- AnnData

These functions are used to validate and normalize the tables
to ensure that conversion between formats is consistent.
"""

from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import polars as pl
from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.utils import NgioTableValidationError, NgioValueError

TabularData = AnnData | DataFrame | PolarsDataFrame | LazyFrame

# -----------------
# Validation utils
# -----------------


def _validate_index_key_df(pandas_df: DataFrame, index_key: str | None) -> DataFrame:
    """Validate the index key of the pandas DataFrame.

    Args:
        pandas_df (DataFrame): The pandas DataFrame to validate.
        index_key (str | None): The column name to use as the index of the DataFrame.

    Returns:
        DataFrame: DataFrame with validated index key.

    Raises:
        NgioTableValidationError: If index key is not found in DataFrame.
    """
    if index_key is None:
        return pandas_df

    if pandas_df.index.name == index_key:
        return pandas_df

    if index_key in pandas_df.columns:
        pandas_df = pandas_df.set_index(index_key)
        pandas_df.index.name = index_key
        return pandas_df

    if pandas_df.index.name is None:
        pandas_df.index.name = index_key
        return pandas_df

    raise NgioTableValidationError(f"Index key '{index_key}' is not found in DataFrame")


def _validate_cast_index_dtype_df(
    pandas_df: DataFrame, index_type: str | None
) -> DataFrame:
    """Check if the index of the DataFrame has the correct dtype.

    Args:
        pandas_df (DataFrame): The pandas DataFrame to validate.
        index_type (str | None): The type to cast the index to ('str' or 'int').

    Returns:
        DataFrame: DataFrame with index of the specified type.

    Raises:
        NgioTableValidationError: If index cannot be cast to the specified type.
        NgioValueError: If index_type is not 'str' or 'int'.
    """
    if index_type is None:
        # Nothing to do
        return pandas_df

    if index_type == "str":
        if ptypes.is_integer_dtype(pandas_df.index):
            # Convert the int index to string is generally safe
            pandas_df = pandas_df.set_index(pandas_df.index.astype(str))

        if not ptypes.is_string_dtype(pandas_df.index):
            raise NgioTableValidationError(
                f"Table index must be of string type, got {pandas_df.index.dtype}"
            )

    elif index_type == "int":
        if ptypes.is_string_dtype(pandas_df.index):
            # Try to convert the string index to int
            try:
                pandas_df = pandas_df.set_index(pandas_df.index.astype(int))
            except ValueError as e:
                if "invalid literal for int() with base 10" in str(e):
                    raise NgioTableValidationError(
                        "Table index must be of integer type, got str."
                        f" We tried implicit conversion and failed: {e}"
                    ) from None
                else:
                    raise e from e

        if not ptypes.is_integer_dtype(pandas_df.index):
            raise NgioTableValidationError(
                f"Table index must be of integer type, got {pandas_df.index.dtype}"
            )
    else:
        raise NgioValueError(
            f"Invalid index type '{index_type}'. Must be 'int' or 'str'."
        )

    return pandas_df


def _check_for_mixed_types(series: pd.Series) -> None:
    """Check if the column has mixed types.

    Args:
        series (pd.Series): The pandas Series to check.

    Raises:
        NgioTableValidationError: If the column has mixed types.
    """
    if series.apply(type).nunique() > 1:  # type: ignore (type lint fails here)
        raise NgioTableValidationError(
            f"Column {series.name} has mixed types: "
            f"{series.apply(type).unique()}. "  # type: ignore (type lint fails here)
            "Type of all elements must be the same."
        )


def _check_for_supported_types(series: pd.Series) -> Literal["str", "int", "numeric"]:
    """Check if the column has supported types.

    Args:
        series (pd.Series): The pandas Series to check.

    Returns:
        Literal["str", "int", "numeric"]: The type category of the series.

    Raises:
        NgioTableValidationError: If the column has unsupported types.
    """
    if ptypes.is_string_dtype(series):
        return "str"
    if ptypes.is_integer_dtype(series):
        return "int"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    raise NgioTableValidationError(
        f"Column {series.name} has unsupported type: {series.dtype}."
        " Supported types are string and numerics."
    )


# -----------------
# Normalization functions
# -----------------


def normalize_pandas_df(
    pandas_df: DataFrame,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    reset_index: bool = False,
) -> DataFrame:
    """Make sure the DataFrame has the correct index and dtype.

    Args:
        pandas_df (DataFrame): The pandas DataFrame to validate.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.
        reset_index (bool): If True the index will be reset (i.e. the index will be
            converted to a column). If False, the index will be kept as is.

    Returns:
        DataFrame: Normalized pandas DataFrame.
    """
    pandas_df = _validate_index_key_df(pandas_df, index_key)
    pandas_df = _validate_cast_index_dtype_df(pandas_df, index_type)
    if pandas_df.index.name is not None:
        index_key = str(pandas_df.index.name)

    if reset_index and pandas_df.index.name is not None:
        pandas_df = pandas_df.reset_index()
    return pandas_df


def normalize_polars_lf(
    polars_lf: LazyFrame | PolarsDataFrame,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
) -> LazyFrame:
    """Validate the polars LazyFrame.

    Args:
        polars_lf (LazyFrame | PolarsDataFrame): The polars LazyFrame to validate.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.

    Returns:
        LazyFrame: Normalized polars LazyFrame.

    Raises:
        ValueError: If index_key is not found or index_type is invalid.
    """
    if index_key is not None:
        schema = polars_lf.collect_schema()
        if index_key not in schema:
            raise NgioTableValidationError(
                f"Index key '{index_key}' not found in LazyFrame columns."
            )

        if index_type is not None:
            if index_type not in ["int", "str"]:
                raise NgioTableValidationError(
                    f"Invalid index type '{index_type}'. Must be 'int' or 'str'."
                )
            if index_type == "int" and not schema[index_key].is_integer():
                polars_lf = polars_lf.with_columns(pl.col(index_key).cast(pl.Int64))
            elif index_type == "str" and not schema[index_key] == pl.String():
                polars_lf = polars_lf.with_columns(pl.col(index_key).cast(pl.String()))

    if isinstance(polars_lf, PolarsDataFrame):
        polars_lf = polars_lf.lazy()
    return polars_lf


def normalize_anndata(
    anndata: AnnData,
    index_key: str | None = None,
) -> AnnData:
    """Validate the AnnData object.

    Args:
        anndata (AnnData): The AnnData object to validate.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.

    Returns:
        AnnData: Normalized AnnData object.
    """
    if index_key is None:
        return anndata
    obs = _validate_index_key_df(anndata.obs, index_key)
    obs = _validate_cast_index_dtype_df(obs, "str")

    if obs.equals(anndata.obs):
        return anndata

    anndata = deepcopy(anndata)
    anndata.obs = obs
    return anndata


# -----------------
# Conversion functions
# -----------------


def convert_pandas_to_polars(
    pandas_df: DataFrame,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
) -> LazyFrame:
    """Convert a pandas DataFrame to a polars LazyFrame.

    Args:
        pandas_df (DataFrame): The pandas DataFrame to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.

    Returns:
        LazyFrame: Converted and normalized polars LazyFrame.
    """
    pandas_df = normalize_pandas_df(
        pandas_df,
        index_key=index_key,
        index_type=index_type,
        reset_index=True,
    )
    return pl.from_pandas(pandas_df).lazy()


def convert_polars_to_pandas(
    polars_df: PolarsDataFrame | LazyFrame,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    reset_index: bool = False,
) -> DataFrame:
    """Convert a polars DataFrame or LazyFrame to a pandas DataFrame.

    Args:
        polars_df (PolarsDataFrame | LazyFrame): The polars DataFrame or
            LazyFrame to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.
        reset_index (bool): If True the index will be reset (i.e., the index will be
            converted to a column). If False, the index will be kept as is.

    Returns:
        DataFrame: Converted and normalized pandas DataFrame.
    """
    if isinstance(polars_df, LazyFrame):
        polars_df = polars_df.collect()

    pandas_df = polars_df.to_pandas()
    pandas_df = normalize_pandas_df(
        pandas_df,
        index_key=index_key,
        index_type=index_type,
        reset_index=reset_index,
    )
    return pandas_df


def convert_pandas_to_anndata(
    pandas_df: DataFrame,
    index_key: str | None = None,
) -> AnnData:
    """Convert a pandas DataFrame to an AnnData object.

    Args:
        pandas_df (DataFrame): The pandas DataFrame to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.

    Returns:
        AnnData: Converted AnnData object.
    """
    pandas_df = normalize_pandas_df(
        pandas_df,
        index_key=index_key,
        index_type="str",
        reset_index=False,
    )

    str_columns, int_columns, num_columns = [], [], []
    for col_name in pandas_df.columns:
        column = pandas_df[col_name]
        _check_for_mixed_types(column)  # Mixed types are not allowed in the table
        col_type = _check_for_supported_types(
            column
        )  # Only string and numeric types are allowed

        if col_type == "str":
            str_columns.append(col_name)

        elif col_type == "int":
            int_columns.append(col_name)

        elif col_type == "numeric":
            num_columns.append(col_name)

    # Converting all observations to string
    obs_df = pandas_df[str_columns + int_columns]
    obs_df.index = pandas_df.index

    x_df = pandas_df[num_columns]

    if x_df.dtypes.nunique() > 1:
        x_df = x_df.astype("float64")

    if x_df.empty:
        # If there are no numeric columns, create an empty array
        # to avoid AnnData failing to create the object
        x_df = np.zeros((len(obs_df), 0), dtype="float64")

    return AnnData(X=x_df, obs=obs_df)


def convert_anndata_to_pandas(
    anndata: AnnData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    reset_index: bool = False,
) -> DataFrame:
    """Convert an AnnData object to a pandas DataFrame.

    Args:
        anndata (AnnData): An AnnData object to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.
        reset_index (bool): If True the index will be reset (i.e., the index will be
            converted to a column). If False, the index will be kept as is.

    Returns:
        DataFrame: Converted and normalized pandas DataFrame.
    """
    pandas_df = anndata.to_df()
    pandas_df[anndata.obs_keys()] = anndata.obs
    pandas_df = normalize_pandas_df(
        pandas_df,
        index_key=index_key,
        index_type=index_type,
        reset_index=reset_index,
    )
    return pandas_df


def convert_anndata_to_polars(
    anndata: AnnData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
) -> LazyFrame:
    """Convert an AnnData object to a polars LazyFrame.

    Args:
        anndata (AnnData): An AnnData object to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.

    Returns:
        LazyFrame: Converted and normalized polars LazyFrame.
    """
    pandas_df = convert_anndata_to_pandas(
        anndata,
        index_key=index_key,
        index_type=index_type,
        reset_index=True,
    )
    return pl.from_pandas(pandas_df).lazy()


def convert_polars_to_anndata(
    polars_df: LazyFrame | PolarsDataFrame,
    index_key: str | None = None,
) -> AnnData:
    """Convert a polars LazyFrame or DataFrame to an AnnData object.

    Args:
        polars_df (LazyFrame | PolarsDataFrame): The polars LazyFrame or
            DataFrame to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.

    Returns:
        AnnData: Converted AnnData object.
    """
    if isinstance(polars_df, LazyFrame):
        polars_df = polars_df.collect()
    pandas_df = polars_df.to_pandas()
    return convert_pandas_to_anndata(
        pandas_df,
        index_key=index_key,
    )


# -----------------
# Conversion functions
# -----------------


def normalize_table(
    table_data: TabularData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
) -> TabularData:
    """Normalize a table to a specific format.

    Args:
        table_data (TabularData): The table to normalize.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.

    Returns:
        DataFrame | AnnData | PolarsDataFrame | LazyFrame: Normalized table.
    """
    if isinstance(table_data, DataFrame):
        return normalize_pandas_df(
            table_data,
            index_key=index_key,
            index_type=index_type,
            reset_index=False,
        )
    if isinstance(table_data, AnnData):
        return normalize_anndata(table_data, index_key=index_key)
    if isinstance(table_data, PolarsDataFrame) or isinstance(table_data, LazyFrame):
        return normalize_polars_lf(
            table_data,
            index_key=index_key,
            index_type=index_type,
        )
    raise NgioValueError(f"Unsupported table type: {type(table_data)}")


def convert_to_anndata(
    table_data: TabularData,
    index_key: str | None = None,
) -> AnnData:
    """Convert a table to an AnnData object.

    Args:
        table_data (TabularData): The table to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.

    Returns:
        AnnData: Converted AnnData object.
    """
    if isinstance(table_data, AnnData):
        return normalize_anndata(table_data, index_key=index_key)
    if isinstance(table_data, DataFrame):
        return convert_pandas_to_anndata(table_data, index_key=index_key)
    if isinstance(table_data, PolarsDataFrame) or isinstance(table_data, LazyFrame):
        return convert_polars_to_anndata(table_data, index_key=index_key)
    raise NgioValueError(f"Unsupported table type: {type(table_data)}")


def convert_to_pandas(
    table_data: TabularData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    reset_index: bool = False,
) -> DataFrame:
    """Convert a table to a pandas DataFrame.

    Args:
        table_data (TabularData): The table to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.
        reset_index (bool): If True the index will be reset (i.e., the index will be
            converted to a column). If False, the index will be kept as is.

    Returns:
        DataFrame: Converted pandas DataFrame.
    """
    if isinstance(table_data, DataFrame):
        return normalize_pandas_df(
            table_data,
            index_key=index_key,
            index_type=index_type,
            reset_index=reset_index,
        )
    if isinstance(table_data, AnnData):
        return convert_anndata_to_pandas(
            table_data,
            index_key=index_key,
            index_type=index_type,
            reset_index=reset_index,
        )
    if isinstance(table_data, PolarsDataFrame) or isinstance(table_data, LazyFrame):
        return convert_polars_to_pandas(
            table_data,
            index_key=index_key,
            index_type=index_type,
            reset_index=reset_index,
        )
    raise NgioValueError(f"Unsupported table type: {type(table_data)}")


def convert_to_polars(
    table_data: TabularData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
) -> LazyFrame:
    """Convert a table to a polars LazyFrame.

    Args:
        table_data (TabularData): The table to convert.
        index_key (str | None): The column name to use as the index of the DataFrame.
            Default is None.
        index_type (str | None): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is None.

    Returns:
        LazyFrame: Converted polars LazyFrame.
    """
    if isinstance(table_data, PolarsDataFrame) or isinstance(table_data, LazyFrame):
        return normalize_polars_lf(
            table_data,
            index_key=index_key,
            index_type=index_type,
        )
    if isinstance(table_data, DataFrame):
        return convert_pandas_to_polars(
            table_data,
            index_key=index_key,
            index_type=index_type,
        )
    if isinstance(table_data, AnnData):
        return convert_anndata_to_polars(
            table_data,
            index_key=index_key,
            index_type=index_type,
        )
    raise NgioValueError(f"Unsupported table type: {type(table_data)}")
