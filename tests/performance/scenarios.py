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

import dask.array as da
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


def _image(ctx, fixture, path="1", **kwargs):
    return open_image(ctx.store(fixture), path=path, mode="r", **kwargs)


def _plate(ctx, fixture="plate", **kwargs):
    return open_ome_zarr_plate(ctx.store(fixture), mode="r", **kwargs)


#: The geometry of the dask-write scenarios. Owned here rather than shared with
#: the image fixture, which is unsharded and must stay that way: these scenarios
#: exist because a *sharded* target is the only shape that can see a partial
#: write unit, and the sharded/unsharded pair is only a clean A/B if nothing
#: else differs between them. Both run on NGFF 0.5 -- sharding needs zarr v3.
#:
#: 128 chunks in 2 shards, so a patch blocked at the chunk shape puts 64 blocks
#: in each shard: 64 chances to read-modify-write it, against the 1 write the
#: shard actually needs.
_WRITE_SHAPE = (1, 8, 256, 256)
_WRITE_CHUNKS = (1, 1, 64, 64)
_WRITE_SHARDS = (1, 4, 256, 256)


def _write_target(ctx, name, shards=None, levels=1):
    """An empty image, ready to be written into.

    Built per scenario in `setup` like `_consolidation_target`, so the
    measurement covers only the write and repeated runs cannot accumulate state.
    """
    container = create_empty_ome_zarr(
        store=ctx.scratch(name),
        shape=_WRITE_SHAPE,
        axes_names=["c", "z", "y", "x"],
        channels_meta=["Channel 1"],
        levels=levels,
        pixelsize=(0.65, 0.65),
        chunks=_WRITE_CHUNKS,
        shards=shards,
        ngff_version="0.5",
        compressors=None,
        overwrite=True,
    )
    return container.get_image(path="0")


def _write_patch(shape=_WRITE_SHAPE):
    # Blocked at the *chunk* shape, which is what a caller who rechunked to
    # `array.chunks` -- or who never thought about it -- ends up with.
    return da.from_array(np.ones(shape, dtype=np.uint16), chunks=_WRITE_CHUNKS)


def _sharded_consolidation_target(ctx):
    """A written sharded pyramid, ready to consolidate.

    The level-0 write is numpy on purpose: it happens in `setup`, and going
    through dask here would put the write path's own amplification into a
    scenario meant to measure the consolidation's.
    """
    image = _write_target(ctx, "consolidate_sharded", shards=_WRITE_SHARDS, levels=3)
    image.set_array(patch=np.ones(_WRITE_SHAPE, dtype=np.uint16))
    return image


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
    # `cache=True` must cost strictly fewer reads than `cache=False`. These two
    # were byte-for-byte identical until `load_attrs` started honouring the
    # cache instead of reopening the group unconditionally; their *inequality*
    # is what holds that fix.
    "open_container": Scenario(None, lambda ctx: _container(ctx, "image", cache=False)),
    "open_container_cached": Scenario(
        None, lambda ctx: _container(ctx, "image", cache=True)
    ),
    # Paired with `open_container`: v0.4 attrs decode on the first registry
    # entry, v0.5 attrs only after one failed pydantic validation.
    "open_container_v04": Scenario(
        None, lambda ctx: _container(ctx, "image_v04", cache=False)
    ),
    # Pure metadata access, and the property iterators touch once per ROI --
    # twice per ROI for a masked one. Uncached it is one full metadata reload
    # per access and must stay linear; cached it must be flat, and the pair is
    # what states which of the two is being measured.
    "dimensions_x10": Scenario(
        lambda ctx: _image(ctx, "image"),
        lambda image: [image.dimensions for _ in range(10)],
    ),
    "dimensions_x10_cached": Scenario(
        lambda ctx: _image(ctx, "image", cache=True),
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
    # Selecting channels by label. Resolving a label needs the channel metadata,
    # and reading that off the image is a full metadata reload -- so this must
    # cost one read regardless of how many channels are named, not one each.
    # `channel_selection=None` short-circuits, so only the selecting path is
    # measured here.
    "read_channel_selection": Scenario(
        lambda ctx: _image(ctx, "image"),
        lambda image: image.get_as_numpy(channel_selection=["Channel 1", "Channel 2"]),
    ),
    # The same selection three times on one image. The channel metadata is
    # cached against the metadata generation, so the metadata cost must be
    # flat in the call count -- only the chunk reads may scale. A per-ROI
    # loop with a `channel_selection` is the ordinary shape of this.
    "read_channel_selection_x3": Scenario(
        lambda ctx: _image(ctx, "image"),
        lambda image: [
            image.get_as_numpy(channel_selection=["Channel 1", "Channel 2"])
            for _ in range(3)
        ],
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
    # The same walk with caching on. `images_paths` reads the plate document
    # once and then one well document per well; with `cache=False` every one of
    # those is re-read several times over. The gap between this and the scenario
    # above is the whole point of the metadata cache, and on a 384-well plate on
    # S3 it is the difference between seconds and minutes.
    "plate_images_paths_cached": Scenario(
        lambda ctx: _plate(ctx, cache=True),
        lambda plate: plate.images_paths(),
    ),
    # Enumerating one well's images. The plate metadata already holds the well
    # path and the well holds its own image list, so this should cost one of
    # each -- not one plate read per image, which is what resolving the prefix
    # per image used to cost. Six images in one well; every other plate fixture
    # has one, which cannot show the difference.
    "plate_well_images_paths": Scenario(
        lambda ctx: _plate(ctx, "plate_multi_image"),
        lambda plate: plate.well_images_paths(row="A", column="01"),
    ),
    # The same walk, then opening each container. Sits on top of the scenario
    # above, so a regression there lands here too.
    "plate_get_well_images": Scenario(
        lambda ctx: _plate(ctx, "plate_multi_image"),
        lambda plate: plate.get_well_images(row="A", column="01"),
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
    # The pyarrow backends read their *payload* against the filesystem directly,
    # so no store counter sees those bytes -- but the zarr metadata around the
    # table is still ordinary store IO, and the group reopens behind it are what
    # `table_load_parquet` holds. csv is absent as a duplicate of parquet: same
    # backend, same code path.
    "table_load_parquet": Scenario(
        lambda ctx: _container(ctx, "tables_parquet"),
        lambda c: c.get_table("features_parquet").dataframe,
    ),
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
    # Filtering by type is the expensive listing: the type lives in each
    # table's own attributes, never in the `/tables` group, so it costs one
    # group open per table. Asking for two types used to walk the whole set
    # twice; the second call must now be nearly free, and the repeat free.
    "list_roi_tables": Scenario(
        lambda ctx: _container(ctx, "tables"),
        lambda c: c.list_roi_tables(),
    ),
    "list_roi_tables_repeated": Scenario(
        lambda ctx: _container(ctx, "tables"),
        lambda c: [c.list_roi_tables() for _ in range(3)],
    ),
    # An image with no `/tables` at all. Answering "nothing" was the *most*
    # expensive listing per call, because the failed probe was never
    # remembered — so this must be flat in the number of calls.
    "list_tables_absent_x3": Scenario(
        lambda ctx: _container(ctx, "image_no_tables"),
        lambda c: [c.list_tables() for _ in range(3)],
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
    # --- writes: dask ------------------------------------------------------
    # The write unit of a zarr array is `shards or chunks`, never `chunks`:
    # zarr can only skip the read-modify-write when a write covers a *whole*
    # unit. `da.store` issues one `__setitem__` per dask block, so on a sharded
    # target a block covering one inner chunk makes zarr read, decode, merge,
    # re-encode and rewrite the entire shard.
    #
    # Read the pair. The unsharded control is the one that must not move; the
    # sharded one is where `get.chunk` and `bytes.read` should be *zero*, and
    # zero is a guarantee rather than a speedup -- no read-modify-write means no
    # lost update, on any machine at any worker count. Every other fixture in
    # this suite is unsharded, so without these the amplification is invisible.
    "write_dask": Scenario(
        lambda ctx: (_write_target(ctx, "write_dask"), _write_patch()),
        lambda state: state[0].set_array(patch=state[1]),
    ),
    "write_sharded_dask": Scenario(
        lambda ctx: (
            _write_target(ctx, "write_sharded_dask", shards=_WRITE_SHARDS),
            _write_patch(),
        ),
        lambda state: state[0].set_array(patch=state[1]),
    ),
    # A region write that covers neither shard wholly, which is the case a
    # caller writing per-ROI actually hits. Its boundary shards are still
    # read-modify-written -- that part is inherent to the geometry, not a defect
    # -- so what this holds is that each of them is read *once*.
    "write_sharded_roi_dask": Scenario(
        lambda ctx: (
            _write_target(ctx, "write_sharded_roi_dask", shards=_WRITE_SHARDS),
            _write_patch((1, 6, 256, 256)),
        ),
        lambda state: state[0].set_array(patch=state[1], z=slice(1, 7)),
    ),
    # --- writes: pyramid consolidation ------------------------------------
    # The most expensive operation in the library, and every writing iterator
    # triggers it implicitly via `post_consolidate`.
    # `auto` landed on this branch and resolves to numpy on this geometry
    # (small, dyadic, linear), so its counts must match `consolidate_numpy` --
    # the resolution itself must not cost a single extra store op.
    **{
        f"consolidate_{mode}": Scenario(
            lambda ctx, mode=mode: _consolidation_target(ctx, mode),
            lambda image, mode=mode: image.consolidate(mode=mode),
        )
        for mode in ("dask", "numpy", "coarsen", "auto")
    },
    # The dask consolidation onto a *sharded* pyramid. `_pyramid.py` rechunks
    # the zoomed level to `target.chunks`, which on a sharded array is the
    # inner chunk shape -- so every block is a partial shard write. Pairs with
    # `consolidate_dask` the same way `write_sharded_dask` pairs with
    # `write_dask`: the unsharded one must not move.
    "consolidate_sharded_dask": Scenario(
        lambda ctx: _sharded_consolidation_target(ctx),
        lambda image: image.consolidate(mode="dask"),
    ),
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
