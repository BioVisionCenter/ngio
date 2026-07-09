"""Round-trip tests for empty tables (issue #99)."""

from pathlib import Path

import pandas as pd
import pytest

from ngio.tables import (
    ConditionTable,
    FeatureTable,
    MaskingRoiTable,
    RoiTable,
)
from ngio.tables._tables_container import open_table, write_table
from ngio.tables.v1 import GenericTable
from ngio.tables.v1._roi_table import REQUIRED_COLUMNS


@pytest.mark.parametrize("backend", ["json", "anndata"])
@pytest.mark.parametrize("table_cls", [RoiTable, MaskingRoiTable])
def test_empty_roi_table_roundtrip(tmp_path: Path, backend: str, table_cls):
    store = tmp_path / "t.zarr"
    write_table(store=store, table=table_cls(), backend=backend)

    loaded = open_table(store=store)
    assert isinstance(loaded, table_cls)
    assert loaded.rois() == []

    df = loaded.dataframe
    assert len(df) == 0
    # The ROI schema is preserved even when the table is empty.
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


@pytest.mark.parametrize("backend", ["json", "anndata"])
def test_empty_feature_table_roundtrip(tmp_path: Path, backend: str):
    store = tmp_path / "t.zarr"
    table = FeatureTable(pd.DataFrame({"label": [], "feat": []}))
    write_table(store=store, table=table, backend=backend)

    loaded = open_table(store=store)
    assert isinstance(loaded, FeatureTable)
    df = loaded.dataframe
    assert len(df) == 0
    assert "feat" in df.columns


@pytest.mark.parametrize("backend", ["json", "anndata"])
@pytest.mark.parametrize("table_cls", [ConditionTable, GenericTable])
def test_empty_generic_table_roundtrip(tmp_path: Path, backend: str, table_cls):
    store = tmp_path / "t.zarr"
    write_table(store=store, table=table_cls(pd.DataFrame()), backend=backend)

    loaded = open_table(store=store)
    assert isinstance(loaded, table_cls)
    assert len(loaded.dataframe) == 0
