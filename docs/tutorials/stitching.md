---
description: Segment a large image tile by tile and merge the objects split at tile seams.
---

# Stitching

**Segment tile by tile, end with one consistent label image.**

Segmenting a large image in tiles creates two problems: every tile numbers its objects
from 1, and an object crossing a tile boundary comes out as two objects with two ids.
This tutorial runs into both on purpose, then declares `with_stitch()` and lets the
iterator solve them. Concepts and vocabulary: the
[iterators guide](../getting_started/6_iterators.md#stitching-a-tiled-segmentation).

## Step 1: open the OME-Zarr container

```python exec="true" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:plot_helpers"
```

```python exec="true" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:open_container"
```

## Step 2: write a segmentation function

A classic watershed pipeline — the same segmentation function the
[ngio workshop](https://github.com/BioVisionCenter/ngio-workshop) uses on this dataset.
The function knows nothing about tiles or ids — keeping ids consistent across tiles is
the iterator's job, not the function's.

```python exec="true" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:segmentation_fn"
```

## Step 3: segment the tiles independently

Tile the image with `by_grid` and segment each tile on its own. Keeping the ids
distinct takes a per-tile offset (`UniqueLabelsTransform`) you have to wire up
yourself — and even then, every nucleus that crosses a tile boundary is counted twice:

```python exec="true" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:naive_tiling"
```

## Step 4: declare `with_stitch()`

`with_stitch()` replaces all of that bookkeeping. Each tile reads a halo past its edge,
so neighbouring tiles predict the same pixels around a seam; where their objects agree,
the ids are joined, and the survivors are renumbered to a dense `1..N`:

```python exec="true" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:stitch"
```

The object count drops — every nucleus split at a seam is now counted once.

### Plot the results

```python exec="true" html="1" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:overview_figure"
```

Zooming onto a corner where four tiles meet shows what changed — the nucleus sitting on
the corner was split into one fragment per tile:

```python exec="true" html="1" source="material-block" session="stitching"
--8<-- "docs/snippets/tutorials/stitching.py:seam_figure"
```

## Step 5: tune it

`with_stitch()` takes a `StitchConfig`:

```python
from ngio.iterators import StitchConfig

iterator = SegmentationIterator(image, label).with_stitch(
    StitchConfig(iou_threshold=0.5, block_size=50_000)
)
```

`iou_threshold` is how much two tiles must agree before their ids are joined. The
default errs towards leaving an object split rather than merging two that are not — an
over-split label can be fixed downstream, a wrong merge cannot. `block_size` is how
many ids each tile is given, and must exceed the largest count a single tile can
produce.

By default the scratch band arrays live in a transient group inside the output label,
which works under every mapper. `scratch_store` puts them elsewhere — often worth
doing, since labels compress well:

```python
from zarr.storage import MemoryStore

StitchConfig(scratch_store=MemoryStore())
```

That keeps the output store untouched and leaves nothing behind if a run dies. The one
restriction is `ProcessMapper`: a `MemoryStore` pickles by value, so each worker would
bank into a private copy — ngio refuses that rather than losing the predictions
silently.

If a run is interrupted between the map and the resolve, the label holds a valid but
over-split segmentation, and re-running the resolve is safe. An interruption *inside*
the resolve is different: the default `compact=True` renumbers the label in place, so
a kill mid-walk leaves mixed ids — ngio marks the walk before it starts and refuses a
retry loudly rather than silently splitting objects; re-run the map to regenerate the
label. Compaction is also available on its own — `label.relabel_sequential()`
renumbers any label to a dense `1..N`.

## Next steps

- [Iterators guide](../getting_started/6_iterators.md#stitching-a-tiled-segmentation) —
  what counts as evidence for a merge, and how stitching relates to halos and overlap.
- [Distributed processing](distributed_processing.md) — the same stitched run split
  across cluster jobs.
- [Image segmentation](image_segmentation.md) — stitching over a FOV table, and masked
  segmentation.
