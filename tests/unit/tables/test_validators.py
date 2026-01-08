from functools import partial

import pytest
from pandas import DataFrame

from ngio.tables._validators import (
    validate_columns,
    validate_table,
    validate_unique_index,
)
from ngio.utils import NgioTableValidationError

@pytest.mark.skip
def test_validate_columns():
    test_df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    required_columns = ["a", "b"]
    optional_columns = ["c"]
    result = validate_columns(test_df, required_columns, optional_columns)
    assert result.equals(test_df)

    with pytest.raises(NgioTableValidationError):
        validate_columns(test_df, ["a", "b", "d"], optional_columns)

    with pytest.raises(NgioTableValidationError):
        validate_columns(test_df, ["a", "b"], optional_columns=["d"])


def test_validate_index():
    test_df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = validate_unique_index(test_df)
    assert result.equals(test_df)

    test_df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[1, 1, 3])
    with pytest.raises(NgioTableValidationError):
        validate_unique_index(test_df)


def test_validate_table():
    test_df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    required_columns = ["a", "b"]
    optional_columns = ["c"]

    validators = [
        partial(
            validate_columns,
            required_columns=required_columns,
            optional_columns=optional_columns,
        ),
        validate_unique_index,
    ]

    out_df = validate_table(test_df, validators)
    assert out_df.equals(test_df)
