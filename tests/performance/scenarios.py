"""The measured scenarios.

A plain dict rather than a registry: there is exactly one consumer, so a
registry would only be a dict with extra steps.

Each scenario is `(setup, run)`. `setup` receives the fixture directory and
runs *outside* the count, so "open once, then read" attributes only the read.
`run` is what gets counted.

Keep this list short. A scenario earns its place by informing a decision — is
this operation doing more work than it should, and would a regression here be
worth failing a build over. Scenarios that merely produce a number do not.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ngio import (
    create_empty_ome_zarr,
    open_image,
    open_ome_zarr_container,
    open_ome_zarr_plate,
)


class Scenario(NamedTuple):
    """One measured operation."""

    setup: Any | None
    run: Any


def _container(ctx, fixture, **kwargs):
    return open_ome_zarr_container(ctx.store(fixture), mode="r", **kwargs)


def _image(ctx, fixture, path="1"):
    return open_image(ctx.store(fixture), path=path, mode="r")


def _plate(ctx):
    return open_ome_zarr_plate(ctx.store("plate"), mode="r")


def _consolidation_target(ctx, mode):
    """A small written pyramid, ready to consolidate.

    Built per scenario in `setup` so the measurement covers only the
    consolidation itself, and so repeated runs cannot accumulate state.
    """
    container = create_empty_ome_zarr(
        store=ctx.scratch(f"consolidate_{mode}"),
        shape=(1, 4, 256, 256),
        axes_names=["c", "z", "y", "x"],
        channels_meta=["Channel 1"],
        levels=3,
        pixelsize=(0.65, 0.65),
        chunks=(1, 1, 128, 128),
        compressors=None,
        overwrite=True,
    )
    image = container.get_image(path="0")
    image.set_array(patch=np.ones((1, 4, 256, 256), dtype=np.uint16))
    return image


SCENARIOS: dict[str, Scenario] = {
    # --- open and metadata ------------------------------------------------
    # `cache=True` should cost strictly fewer reads than `cache=False`. It
    # currently costs the same, because `ZarrGroupHandler.load_attrs` reopens
    # the group unconditionally; the equality of these two is that bug.
    "open_container": Scenario(None, lambda ctx: _container(ctx, "image", cache=False)),
    "open_container_cached": Scenario(
        None, lambda ctx: _container(ctx, "image", cache=True)
    ),
    # Paired with `open_container`: v0.4 attrs decode on the first registry
    # entry, v0.5 attrs only after one failed pydantic validation.
    "open_container_v04": Scenario(
        None, lambda ctx: _container(ctx, "image_v04", cache=False)
    ),
    # Pure metadata access. Counts should be flat in the number of accesses
    # once the meta handler caches; today they grow linearly.
    "dimensions_x10": Scenario(
        lambda ctx: _image(ctx, "image"),
        lambda image: [image.dimensions for _ in range(10)],
    ),
    # --- reads ------------------------------------------------------------
    "read_full": Scenario(
        lambda ctx: _image(ctx, "image", path="2"),
        lambda image: image.get_as_numpy(),
    ),
    "read_rois": Scenario(
        lambda ctx: _rois(ctx),
        lambda state: [state[0].get_roi_as_numpy(roi) for roi in state[1]],
    ),
    # --- plate ------------------------------------------------------------
    # Plate metadata only; the well count should not appear in the counts.
    "plate_wells_paths": Scenario(_plate, lambda plate: plate.wells_paths()),
    "plate_get_wells": Scenario(_plate, lambda plate: plate.get_wells()),
    # Calls get_wells internally, so this costs one group open per well to
    # answer a question the plate metadata alone could answer.
    "plate_images_paths": Scenario(_plate, lambda plate: plate.images_paths()),
    # --- tables -----------------------------------------------------------
    # csv and parquet are absent on purpose: they go through pyarrow against
    # the filesystem directly, bypassing the zarr store, so no store counter
    # can see them.
    "table_load_anndata": Scenario(
        lambda ctx: _container(ctx, "tables"),
        lambda c: c.get_table("features_anndata_v1").dataframe,
    ),
    "table_load_json": Scenario(
        lambda ctx: _container(ctx, "tables"),
        lambda c: c.get_table("features_experimental_json_v1").dataframe,
    ),
    "table_load_roi": Scenario(
        lambda ctx: _container(ctx, "tables"),
        lambda c: c.get_table("well_ROI_table").rois(),
    ),
    # --- writes: pyramid consolidation ------------------------------------
    # The most expensive operation in the library, and every writing iterator
    # triggers it implicitly via `post_consolidate`.
    #
    # NOTE: the "dask" numbers below record a known 2x waste, not correct
    # behaviour. `_pyramid.py:44` calls `compute_chunk_sizes()` right after an
    # explicit `rechunk(target.chunks)`, which executes the whole read -> zoom
    # graph just to learn block shapes it already knows, then `da.store` runs
    # the same graph again. Removing that line should roughly halve `get.chunk`
    # and `bytes.read` here, and that halving is the point of gating it.
    **{
        f"consolidate_{mode}": Scenario(
            lambda ctx, mode=mode: _consolidation_target(ctx, mode),
            lambda image, mode=mode: image.consolidate(mode=mode),
        )
        for mode in ("dask", "numpy", "coarsen")
    },
}


def _rois(ctx):
    from ngio.iterators._rois_utils import grid

    image = _image(ctx, "image", path="1")
    rois = grid(
        rois=image.build_image_roi_table().rois(),
        ref_image=image,
        size_y=64,
        size_x=64,
    )
    return image, rois
