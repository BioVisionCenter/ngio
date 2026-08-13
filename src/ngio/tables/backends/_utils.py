"""Utility functions for converting between different tables formats.

The supported formats are:
- pandas DataFrame
- polars DataFrame or LazyFrame
- AnnData

These functions are used to validate and normalize the tables
to ensure that conversion between formats is consistent.
"""

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

    if len(pandas_df.index) == 0 and index_type in ("str", "int"):
        # Empty table: the index has no values to validate, so casting always
        # succeeds. This lets empty tables be normalized without a spurious
        # "index must be of X type" error.
        target = str if index_type == "str" else int
        return pandas_df.set_index(pandas_df.index.astype(target))

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

    Missing values (None/NaN) are ignored, so a column of a single type plus
    missing entries is not considered mixed.

    Raises:
        NgioTableValidationError: If the column has mixed types.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        # The categories carry every distinct value, so scan those — O(k)
        # instead of a Python call per element.
        series = pd.Series(series.cat.categories, name=series.name)
    elif series.dtype != object:
        # Any other concrete dtype (int64, float64, bool, datetime, string)
        # already guarantees homogeneity; only `object` can hide mixed Python
        # types. The per-cell scan below is a Python call per element, paid
        # per column on every pandas -> anndata conversion.
        return
    non_null = series.dropna()
    if non_null.apply(type).nunique() > 1:
        raise NgioTableValidationError(
            f"Column {series.name} has mixed types: "
            f"{non_null.apply(type).unique()}. "
            "Type of all elements must be the same."
        )


def _check_for_supported_types(series: pd.Series) -> Literal["str", "int", "numeric"]:
    """Check if the column has supported types.

    Args:
        series (pd.Series): The pandas Series to check.

    Missing values (None/NaN) are ignored: a string column that also contains
    missing entries (or is entirely missing) is still classified as "str", since
    AnnData serializes such columns natively (as a categorical with NaN).

    Returns:
        Literal["str", "int", "numeric"]: The type category of the series.

    Raises:
        NgioTableValidationError: If the column has unsupported types.
    """
    if ptypes.is_integer_dtype(series):
        return "int"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    # String-like columns (object/str/category), possibly with missing values.
    # is_string_dtype is unreliable for object columns that contain None and for
    # the category dtype produced by an AnnData round trip, so classify from the
    # non-null contents instead.
    non_null = series.dropna()
    if non_null.empty or all(isinstance(value, str) for value in non_null):
        return "str"
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
    # anndata>=0.12 types `.obs` as DataFrame | Dataset2D; ngio only ever
    # builds in-memory AnnData, so it is always a DataFrame here.
    obs = _validate_index_key_df(anndata.obs, index_key)  # type: ignore
    obs = _validate_cast_index_dtype_df(obs, "str")

    if obs.equals(anndata.obs):
        return anndata

    # Only `obs` changes, so build a new AnnData around the same `X` (and the
    # other components) instead of deep-copying the whole object — the copy
    # duplicated the full matrix in memory to swap one frame. The components
    # are shared, not copied; nothing downstream mutates them.
    return AnnData(
        X=anndata.X,
        obs=obs,
        # Same `DataFrame | Dataset2D` union as `.obs` above: ngio only ever
        # builds in-memory AnnData, so it is always a DataFrame here.
        var=anndata.var,  # ty: ignore[invalid-argument-type]
        uns=anndata.uns,
        obsm=dict(anndata.obsm),
        varm=dict(anndata.varm),
        layers=dict(anndata.layers),
        obsp=dict(anndata.obsp),
        varp=dict(anndata.varp),
    )


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

    if x_df.shape[1] == 0:
        # If there are no numeric columns, create an empty array
        # to avoid AnnData failing to create the object. Note: we check the number
        # of columns rather than DataFrame.empty, since .empty is also True for a
        # frame with columns but zero rows (an empty table), which must keep its X.
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
    pandas_df[anndata.obs.columns.to_list()] = anndata.obs
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
