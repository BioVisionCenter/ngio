---
description: Run a spot detector tile by tile and store the boxes as a ROI table.
---

# Object detection

**Detect objects tile by tile and store the boxes as one ROI table.**

Not every model produces a segmentation mask. A detector — a YOLO network, a spot
finder — reports **bounding boxes**, and the natural home for those in an OME-Zarr
container is a ROI table. The `ObjectDetectionIterator` runs a detector one tile at a
time and owns everything around it: world coordinates, cross-tile deduplication, and
one table at the end. Here the "model" is a Laplacian-of-Gaussian spot finder from
`skimage`, so the whole tutorial runs without any ML dependency.

## Step 1: write the detector function

The detector sees one tile's pixels and answers with a list of `Roi` boxes in the
tile's own pixel coordinates (`space="pixel"`); the iterator anchors them into world
coordinates. Any extra field — here the peak intensity as a confidence — rides along
into the final table.

```python exec="true" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:detector"
```

## Step 2: create the OME-Zarr image

```python exec="true" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:create"
```

## Step 3: detect, tile by tile

The image is tiled with `by_grid`, and each tile reads a `with_halo` margin past its own
edge — this is the one read-only iterator where the halo is allowed, because there is
no write to crop it from. The margin means a nucleus cut by a tile boundary is seen
whole by at least one tile; the duplicate detections that produces are resolved by
non-maximum suppression, configured with `nms=NmsConfig(...)`. The per-tile detection
fans out under a `mapper` like any `reduce`; nothing is written until your own
`add_table` call.

```python exec="true" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:detect"
```

### Sanity check: read the table back

```python exec="true" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:plot_helpers"
```

```python exec="true" html="1" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:read_table_back"
```

The boxes come back as ordinary `Roi` objects in world coordinates, so drawing them is
one `to_pixel` per box:

```python exec="true" html="1" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:plot_detections"
```

## Next steps

- [Iterators](../getting_started/6_iterators.md) — halos, parallel mappers, and the
  detection contract in full.
- [Feature extraction](feature_extraction.md) — the read-only sibling: measurements
  into a feature table.
- [Table specifications](../table_specs/overview.md) — how ROI tables are stored.
