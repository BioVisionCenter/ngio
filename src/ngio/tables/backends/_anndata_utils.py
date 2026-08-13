from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import zarr
from anndata import AnnData
from anndata._io.specs import read_elem
from anndata._io.utils import _read_legacy_raw
from anndata._io.zarr import read_dataframe
from anndata._settings import settings
from anndata.compat import _clean_uns
from anndata.experimental import read_dispatched

from ngio.utils import (
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)
from ngio.utils._zarr_utils import list_group_keys

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _update_anndata_global_settings(zarr_format: Literal[2, 3]) -> None:
    """Update global settings for anndata's zarr read/write functions.

    This is needed to ensure that anndata uses the correct zarr format when
    reading/writing tables.

    Args:
        zarr_format (Literal[2, 3]): The zarr format version to use.
            Must be either 2 or 3.
    """
    if zarr_format == 2:
        # Added to avoid user issues when writing
        # v2 and v3 in the same session
        # order matters here, we need to set auto_shard_zarr_v3
        # before setting zarr_write_format
        settings.auto_shard_zarr_v3 = False
        settings.zarr_write_format = 2
    else:
        settings.zarr_write_format = 3
        # Added to avoid user warning in anndata 0.12.14
        settings.auto_shard_zarr_v3 = True


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

    # One directory listing answers both "can this store list?" and "which
    # elements exist". Probing an absent element costs a store round-trip
    # (`elem.get` fetches metadata to miss), so on a listable store the
    # candidates are cut to what is actually there.
    keys = list_group_keys(group)
    if keys is None:
        # If not listable we filter some elements
        non_listable_elems = ["uns", "obsm", "varm", "obsp", "varp", "layers"]
        elem_to_read = [elem for elem in elem_to_read if elem not in non_listable_elems]
    else:
        elem_to_read = [elem for elem in elem_to_read if elem in keys]

    # Read with handling for backwards compat
    obs_was_array = False

    def callback(func: Callable, elem_name: str, elem: Any, iospec: Any) -> Any:
        if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
            # Heterogeneous by construction: the keys are AnnData field names
            # and the values whatever `read_dispatched` returns for each.
            ad_kwargs: dict[str, Any] = {}
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
            if elem_name == "/obs" and isinstance(elem, zarr.Array):
                # anndata <0.7 wrote `obs` as an array; remembered here so the
                # compat path below does not re-fetch the node to find out.
                nonlocal obs_was_array
                obs_was_array = True
            return read_dataframe(elem)
        elif elem_name == "/raw":
            # Backwards compat
            return _read_legacy_raw(group, func(elem), read_dataframe, func)
        return func(elem)

    adata = read_dispatched(group, callback=callback)  # type: ignore

    # Backwards compat (should figure out which version). The membership test
    # is a store probe, skipped when the listing already answered it.
    if "raw.X" in keys if keys is not None else "raw.X" in group:
        raw = AnnData(**_read_legacy_raw(group, adata.raw, read_dataframe, read_elem))  # type: ignore
        raw.obs_names = adata.obs_names  # type: ignore
        adata.raw = raw  # type: ignore

    # Backwards compat for <0.7
    if obs_was_array:
        _clean_uns(adata)

    if isinstance(adata, dict):
        adata = AnnData(**adata)  # type: ignore
    if not isinstance(adata, AnnData):
        raise NgioValueError(f"Expected an AnnData object, but got {type(adata)}")
    return adata
