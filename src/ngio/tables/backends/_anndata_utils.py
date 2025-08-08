from __future__ import annotations

from typing import TYPE_CHECKING, Any

import zarr
from anndata import AnnData
from anndata._io.specs import read_elem
from anndata._io.utils import _read_legacy_raw
from anndata._io.zarr import read_dataframe
from anndata.compat import _clean_uns
from anndata.experimental import read_dispatched

from ngio.utils import (
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def custom_anndata_read_zarr(
    store: StoreOrGroup, elem_to_read: Sequence[str] | None = None
) -> AnnData:
    """Read from a hierarchical Zarr array store.

    # Implementation originally from https://github.com/scverse/anndata/blob/main/src/anndata/_io/zarr.py
    # Original implementation would not work with remote storages so we had to copy it
    # here and slightly modified it to work with remote storages.

    Args:
        store (StoreOrGroup): A store or group to read the AnnData from.
        elem_to_read (Sequence[str] | None): The elements to read from the store.
    """
    group = open_group_wrapper(store=store, mode="r")

    if not isinstance(group.store, zarr.DirectoryStore):
        elem_to_read = ["X", "obs", "var"]

    if elem_to_read is None:
        elem_to_read = [
            "X",
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "obsp",
            "varp",
            "layers",
        ]

    # Read with handling for backwards compat
    def callback(func: Callable, elem_name: str, elem: Any, iospec: Any) -> Any:
        if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
            ad_kwargs = {}
            # Some of these elem fail on https
            # So we only include the ones that are strictly necessary
            # for fractal tables
            # This fails on some https
            # base_elem += list(elem.keys())
            for k in elem_to_read:
                v = elem.get(k)
                if v is not None and not k.startswith("raw."):
                    ad_kwargs[k] = read_dispatched(v, callback)  # type: ignore
            return AnnData(**ad_kwargs)

        elif elem_name.startswith("/raw."):
            return None
        elif elem_name in {"/obs", "/var"}:
            return read_dataframe(elem)
        elif elem_name == "/raw":
            # Backwards compat
            return _read_legacy_raw(group, func(elem), read_dataframe, func)
        return func(elem)

    adata = read_dispatched(group, callback=callback)  # type: ignore

    # Backwards compat (should figure out which version)
    if "raw.X" in group:
        raw = AnnData(**_read_legacy_raw(group, adata.raw, read_dataframe, read_elem))  # type: ignore
        raw.obs_names = adata.obs_names  # type: ignore
        adata.raw = raw  # type: ignore

    # Backwards compat for <0.7
    if isinstance(group["obs"], zarr.Array):
        _clean_uns(adata)

    if not isinstance(adata, AnnData):
        raise NgioValueError(f"Expected an AnnData object, but got {type(adata)}")
    return adata
