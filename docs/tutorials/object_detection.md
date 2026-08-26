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

The image is tiled with `by_grid` plus a `with_halo` read margin — Step 4 shows why;
the full mechanics are in the
[iterators guide](../getting_started/6_iterators.md#detecting-objects-into-a-roi-table).

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

## Step 4: watch NMS work

The halo solves the boundary problem — a nucleus cut by a tile edge is still seen
whole by at least one neighbour — at the price that both neighbours report it.
Non-maximum suppression settles that: of two boxes overlapping at or above
`iou_threshold`, only the higher-scoring one survives. You can watch it happen — the
raw, pre-NMS view is a loop over the same haloed tiles, anchoring each tile's boxes
into world coordinates, which is exactly what `detect` does before suppressing.

```python exec="true" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:nms_raw"
```

Most duplicates coincide exactly — both tiles saw the nucleus whole, reported the same
box, and either can survive. The interesting ones sit near a seam, where one tile saw
the nucleus clipped at its halo's reach: the boxes disagree, and NMS keeps the one the
score ranks higher. Zooming onto a pair of tile boundaries shows both kinds:

```python exec="true" html="1" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:nms_figure"
```

### Anchor a local box yourself

The coordinate bookkeeping is one call you can also use in a custom flow: `Roi.anchor`
turns a region's ROI plus a patch-local pixel box into the absolute world ROI —

```python exec="true" source="material-block" session="object_detection"
--8<-- "docs/snippets/tutorials/object_detection.py:anchor_demo"
```

The `space` fields are what keep this honest: `anchor` refuses a world-space box
(already absolute — anchoring it would double the offset) and a pixel-space region, so
the classic silent frame mix-up raises instead. Axes the box does not pin inherit the
region's extent.

## Next steps

- [Iterators](../getting_started/6_iterators.md#detecting-objects-into-a-roi-table) —
  the detection contract in full.
- [Feature extraction](feature_extraction.md) — the read-only sibling: measurements
  into a feature table.
- [Table specifications](../table_specs/overview.md) — how ROI tables are stored.
