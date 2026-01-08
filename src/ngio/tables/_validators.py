from collections.abc import Iterable
from typing import Protocol
from warnings import warn

import pandas as pd

from ngio.utils import NgioTableValidationError


class TableValidator(Protocol):
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        """Validate the table DataFrame.

        A Validator is just a simple callable that takes a
            DataFrame and returns a DataFrame.

        If the DataFrame is valid, the same DataFrame is returned.
        If the DataFrame is invalid, the Validator can either modify the DataFrame
            to make it valid or raise a NgioTableValidationError.

        Args:
            table (pd.DataFrame): The DataFrame to validate.

        Returns:
            pd.DataFrame: The validated DataFrame.

        """
        ...


def validate_table(
    table_df: pd.DataFrame,
    validators: Iterable[TableValidator] | None = None,
) -> pd.DataFrame:
    """Validate the table DataFrame.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        validators (Collection[Validator] | None): A collection of functions
            used to validate the table. Default is None.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    validators = validators or []

    # Apply all provided validators
    for validator in validators:
        table_df = validator(table_df)

    return table_df


####################################################################################################
#
# Common table validators
#
####################################################################################################
def validate_columns(
    table_df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate the columns headers of the table.

    If a required column is missing, a TableValidationError is raised.
    If a list of optional columns is provided, only required and optional columns are
        allowed in the table.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        required_columns (list[str]): A list of required columns.
        optional_columns (list[str] | None): A list of optional columns.
            Default is None.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    table_header = table_df.columns
    for column in required_columns:
        if column not in table_header:
            warn(
                f"Could not find required column: {column} in the table",
                stacklevel=2,
            )

    if optional_columns is None:
        return table_df

    possible_columns = [*required_columns, *optional_columns]
    for column in table_header:
        if column not in possible_columns:
            warn(
                f"Could not find column: {column} in the list of possible columns. "
                f"Possible columns are: {possible_columns}",
                stacklevel=2,
            )

    return table_df


def validate_unique_index(table_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the index of the table is unique."""
    if table_df.index.is_unique:
        return table_df

    # Find the duplicates
    duplicates = table_df.index[table_df.index.duplicated()].tolist()
    raise NgioTableValidationError(
        f"Index of the table contains duplicates values. Duplicate: {duplicates}"
    )
