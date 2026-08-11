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
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    create_ome_zarr_from_array,
    open_image,
    open_ome_zarr_container,
    open_ome_zarr_plate,
)

#: The well grid of the `create_plate` scenario. Stated here rather than shared
#: with the plate fixture: a scenario owns the shape of what it measures, the
#: same way `_consolidation_target` owns its own create parameters.
_PLATE_IMAGES = [
    ImageInWellPath(row=row, column=f"{col + 1:02d}", path="0")
    for row in ("A", "B", "C", "D")
    for col in range(6)
]


class Scenario(NamedTuple):
    """One measured operation."""

    setup: Any | None
    run: Any


def _container(ctx, fixture, **kwargs):
    return open_ome_zarr_container(ctx.store(fixture), mode="r", **kwargs)


def _image(ctx, fixture, path="1"):
    return open_image(ctx.store(fixture), path=path, mode="r")


def _plate(ctx, fixture="plate"):
    return open_ome_zarr_plate(ctx.store(fixture), mode="r")


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
    # The same walk over a 0.5 plate. The decoder registry is 0.4-first, so
    # every 0.5 document costs a failed pydantic validation before the right
    # decoder runs; the plate and well handlers memoise the resolved version to
    # pay that once per handler instead of once per read. `meta.decode_fail`
    # here is that memo's gate — the 0.4 scenario above cannot see it, since a
    # 0.4 document hits on the first try and never fails.
    "plate_images_paths_v05": Scenario(
        lambda ctx: _plate(ctx, "plate_v05"),
        lambda plate: plate.images_paths(),
    ),
    # --- plate-wide table aggregation --------------------------------------
    # Both open every image container inside the counted block, on top of the
    # enumeration `plate_images_paths` already gates. That multiplication is
    # the point: a regression in `open_container` lands here N times over, and
    # on S3 each of those opens is a round-trip.
    "plate_list_image_tables": Scenario(
        lambda ctx: _plate(ctx, "plate_tables"),
        lambda plate: plate.list_image_tables(),
    ),
    "plate_concatenate_tables": Scenario(
        lambda ctx: _plate(ctx, "plate_tables"),
        lambda plate: plate.concatenate_image_tables(name="features"),
    ),
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
    # --- writes: creation -------------------------------------------------
    # No pixel data, so the counts are the per-level metadata and empty-array
    # writes alone. `set.meta` should scale with `levels`, nothing else should.
    "create_container_empty": Scenario(
        None,
        lambda ctx: create_empty_ome_zarr(
            store=ctx.scratch("create_empty"),
            shape=(1, 4, 256, 256),
            axes_names=["c", "z", "y", "x"],
            channels_meta=["Channel 1"],
            levels=3,
            pixelsize=(0.65, 0.65),
            chunks=(1, 1, 128, 128),
            compressors=None,
            overwrite=True,
        ),
    ),
    # Paired with the above: the difference is the pyramid write and the
    # percentile pass. It therefore overlaps `consolidate_dask` — read the two
    # together, and expect a fix to one to move this too.
    "create_container_from_array": Scenario(
        lambda ctx: (ctx, np.ones((1, 4, 256, 256), dtype=np.uint16)),
        lambda state: create_ome_zarr_from_array(
            store=state[0].scratch("create_from_array"),
            array=state[1],
            axes_names=["c", "z", "y", "x"],
            channels_meta=["Channel 1"],
            levels=3,
            pixelsize=(0.65, 0.65),
            chunks=(1, 1, 128, 128),
            compressors=None,
            overwrite=True,
        ),
    ),
    # Registers 24 images in one metadata write-out. Costs should be flat in
    # the image count, not one round-trip per well.
    "create_plate": Scenario(
        None,
        lambda ctx: create_empty_plate(
            ctx.scratch("create_plate"),
            name="bench_create_plate",
            images=_PLATE_IMAGES,
            overwrite=True,
        ),
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
