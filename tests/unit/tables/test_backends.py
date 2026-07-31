from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from ngio.tables.backends import (
    AnnDataBackend,
    CsvTableBackend,
    ImplementedTableBackends,
    JsonTableBackend,
    ParquetTableBackend,
    convert_anndata_to_pandas,
    convert_pandas_to_anndata,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


def test_backend_manager(tmp_path: Path):
    manager = ImplementedTableBackends()

    assert set(manager.available_backends) == {
        "json",
        "experimental_json_v1",
        "anndata",
        "anndata_v1",
        "csv",
        "experimental_csv_v1",
        "parquet",
        "experimental_parquet_v1",
    }
    manager.add_backend(JsonTableBackend, overwrite=True)

    manager2 = ImplementedTableBackends()
    assert set(manager2.available_backends) == {
        "json",
        "experimental_json_v1",
        "anndata",
        "anndata_v1",
        "csv",
        "experimental_csv_v1",
        "parquet",
        "experimental_parquet_v1",
    }
    # check if singleton pattern works
    assert manager is manager2

    store = tmp_path / "test_backend_manager.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = manager.get_backend(backend_name="json", group_handler=handler)
    assert isinstance(backend, JsonTableBackend)

    backend = manager.get_backend(backend_name="anndata", group_handler=handler)
    assert isinstance(backend, AnnDataBackend)

    with pytest.raises(NgioValueError):
        manager.get_backend(group_handler=handler, backend_name="non_existent_backend")

    with pytest.raises(NgioValueError):
        manager.add_backend(JsonTableBackend)


@pytest.mark.parametrize(
    "backend_cls, backend_name, implements_anndata, handler_kwargs, backend_kwargs",
    [
        (JsonTableBackend, "json", False, {}, {"index_type": "str"}),
        (CsvTableBackend, "csv", False, {}, {}),
        (ParquetTableBackend, "parquet", False, {}, {}),
        (AnnDataBackend, "anndata", True, {"zarr_format": 2}, {"index_type": "int"}),
    ],
)
def test_backend_round_trip(
    tmp_path: Path,
    backend_cls: type,
    backend_name: str,
    implements_anndata: bool,
    handler_kwargs: dict,
    backend_kwargs: dict,
):
    store = tmp_path / f"test_{backend_name}_backend.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a", **handler_kwargs)
    backend = backend_cls()
    backend.set_group_handler(handler, **backend_kwargs)

    assert backend.backend_name() == backend_name
    assert backend.implements_anndata() == implements_anndata
    assert backend.implements_pandas()

    test_table = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4.1, 5.1, 6.1], "c": ["a", "b", "c"]}
    )
    if backend_kwargs.get("index_type") == "str":
        test_table.index = test_table.index.astype(str)

    backend.write(test_table, metadata={"test": "test"})
    loaded_table = backend.load_as_pandas_df()

    # The anndata round trip does not preserve the index, compare columns only
    for column in loaded_table.columns:
        pd.testing.assert_series_equal(loaded_table[column], test_table[column])
    if not implements_anndata:
        assert loaded_table.equals(test_table), loaded_table

    meta = backend._group_handler.load_attrs()
    assert meta["test"] == "test"
    assert meta["backend"] == backend_name

    a_data = backend.load_as_anndata()
    if implements_anndata:
        backend.write(a_data, metadata={"test": "test"})
        # Test that the "raw" entry with encoding-type "null" is
        # removed after writing for compatibility with older anndata versions
        with pytest.raises(KeyError):
            backend._group_handler._group["raw"]
    else:
        with pytest.raises(NotImplementedError):
            backend.write(a_data, metadata={"test": "test"})

    lf_data = backend.load_as_polars_lf()
    backend.write(lf_data, metadata={"test": "test"})


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_anndata_to_dataframe(index_label: str | None, index_type: str):
    test_obs = pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]})
    test_x = pd.DataFrame({"b": [4.0, 5.0, 6.0]})

    if index_label is None:
        test_obs.index = test_obs.index.astype(str)
    else:
        test_obs.index = pd.Index(["1", "2", "3"], name=index_label)

    anndata = ad.AnnData(obs=test_obs, X=test_x)

    dataframe = convert_anndata_to_pandas(
        anndata,
        index_key=index_label,
        index_type=index_type,  # type: ignore[arg-type]
    )

    for column in test_obs.columns:
        pd.testing.assert_series_equal(
            dataframe[column], test_obs[column], check_index=False
        )

    for column in test_x.columns:
        pd.testing.assert_series_equal(
            dataframe[column], test_x[column], check_index=False
        )

    if index_label is not None:
        assert dataframe.index.name == index_label
        if index_type == "int":
            assert ptypes.is_integer_dtype(dataframe.index)
        elif index_type == "str":
            assert ptypes.is_string_dtype(dataframe.index)


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_dataframe_to_anndata(index_label: str | None, index_type: str):
    test_table = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [4.0, 5.0, 6.0],
        }
    )

    if index_label is not None and index_type == "int":
        test_table.index = pd.Index([1, 2, 3], name=index_label)
    elif index_label is not None and index_type == "str":
        test_table.index = pd.Index(["a", "b", "c"], name=index_label)

    anndata = convert_pandas_to_anndata(
        test_table,
        index_key=index_label,
    )

    for column in anndata.obs.columns:
        pd.testing.assert_series_equal(
            anndata.obs[column], test_table[column], check_index=False
        )

    if index_label is not None:
        assert anndata.obs.index.name == index_label

    for i, column in enumerate(anndata.var.index):
        np.testing.assert_allclose(anndata.X[:, i], test_table[column].values)  # type: ignore


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_round_trip(index_label: str | None, index_type: str):
    test_table = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [4.0, 5.0, 6.0],
        }
    )

    if index_label is not None and index_type == "int":
        test_table.index = pd.Index([1, 2, 3], name=index_label)
    elif index_label is not None and index_type == "str":
        test_table.index = pd.Index(["a", "b", "c"], name=index_label)

    anndata = convert_pandas_to_anndata(
        test_table,
        index_key=index_label,
    )
    datafame = convert_anndata_to_pandas(
        anndata,
        index_key=index_label,
        index_type=index_type,  # type: ignore[arg-type]
    )

    for column in datafame.columns:
        pd.testing.assert_series_equal(
            datafame[column], test_table[column], check_index=True
        )


def _retry_policy():
    from ngio.config import ConstantBackoff, RetryConfig

    return RetryConfig(
        max_retries=2,
        retry_on=["OSError"],
        backoff=ConstantBackoff(delay_s=0.0, jitter=False),
    )


def test_parquet_backend_retries_flaky_load(tmp_path: Path, monkeypatch):
    import ngio.tables.backends._py_arrow_backends as pab
    from ngio.config import get_config

    store = tmp_path / "test_parquet_retry.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = ParquetTableBackend()
    backend.set_group_handler(handler)
    test_table = pd.DataFrame({"a": [1, 2, 3]})
    backend.write(test_table, metadata={})

    monkeypatch.setattr(get_config(), "io_retry", _retry_policy())
    calls = {"n": 0}
    real_dataset = pab.pa_ds.dataset

    def flaky_dataset(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("flaky parquet read")
        return real_dataset(*args, **kwargs)

    monkeypatch.setattr(pab.pa_ds, "dataset", flaky_dataset)
    loaded = backend.load_as_pandas_df()
    assert loaded["a"].tolist() == [1, 2, 3]
    assert calls["n"] == 2


def test_parquet_backend_retries_flaky_write(tmp_path: Path, monkeypatch):
    import ngio.tables.backends._py_arrow_backends as pab
    from ngio.config import get_config

    store = tmp_path / "test_parquet_retry_write.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = ParquetTableBackend()
    backend.set_group_handler(handler)

    monkeypatch.setattr(get_config(), "io_retry", _retry_policy())
    calls = {"n": 0}
    real_write = pab.pa_parquet.write_table

    def flaky_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("flaky parquet write")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(pab.pa_parquet, "write_table", flaky_write)
    backend.write(pd.DataFrame({"a": [1, 2, 3]}), metadata={})
    assert calls["n"] == 2
    assert backend.load_as_pandas_df()["a"].tolist() == [1, 2, 3]


def test_anndata_backend_retries_flaky_write(tmp_path: Path, monkeypatch):
    from ngio.config import get_config

    store = tmp_path / "test_anndata_retry.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = AnnDataBackend()
    backend.set_group_handler(handler)
    obs = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["x", "y", "z"]))
    table = ad.AnnData(obs=obs)

    monkeypatch.setattr(get_config(), "io_retry", _retry_policy())
    calls = {"n": 0}
    real_write = ad.AnnData.write_zarr

    def flaky_write(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("flaky anndata write")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "write_zarr", flaky_write)
    backend.write(table, metadata={})
    assert calls["n"] == 2
