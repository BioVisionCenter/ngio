from pathlib import Path

import pandas as pd
import pytest

from ngio.tables import ConditionTable
from ngio.tables._tables_container import (
    open_table,
    open_tables_container,
    write_table,
)


@pytest.mark.parametrize("backend", ["json", "anndata"])
def test_condition_table_roundtrip(tmp_path: Path, backend: str):
    store = tmp_path / "test.zarr"
    df = pd.DataFrame({"condition": ["cond_a", "cond_b"]})
    write_table(store=store, table=ConditionTable(table_data=df), backend=backend)

    loaded_table = open_table(store=store)
    assert isinstance(loaded_table, ConditionTable)
    assert set(loaded_table.dataframe.columns) == {"condition"}


@pytest.mark.parametrize("source_backend", ["json", "anndata"])
def test_condition_table_copy_with_none(tmp_path: Path, source_backend: str):
    """Regression: copying a condition table with a None entry must not fail.

    Mirrors the fractal `apply_registration_to_image` copy path
    (`get_table` + `add_table`, which delegate to `TablesContainer.get`/`add`):
    a table containing a None is written, reopened, then re-written through the
    default anndata backend.
    """
    src = open_tables_container(tmp_path / "source.zarr", mode="a")
    dst = open_tables_container(tmp_path / "dest.zarr", mode="a")

    df = pd.DataFrame({"condition": ["cond_a", None]})
    src.add(name="condition", table=ConditionTable(table_data=df), backend=source_backend)

    # get_table + add_table analogue (add re-writes with the default anndata backend).
    copied_table = src.get("condition")
    dst.add(name="condition", table=copied_table)

    condition = dst.get("condition").dataframe["condition"]
    assert condition.tolist()[0] == "cond_a"
    assert pd.isna(condition).tolist() == [False, True]
