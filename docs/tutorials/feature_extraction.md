---
description: Extract regionprops features and store them as an ngio feature table.
---

# Feature extraction

**Measure per-label features and store them as a table.**

Measure regionprops features from a segmented image with `ngio` and `skimage`, and write
them back as a feature table in the OME-Zarr container. By the end the container holds a
table with one row per label, ready to be read back or aggregated across a plate.

## Step 1: write the measurement function

Start with the function that does the measuring — here a thin wrapper around
`skimage.measure.regionprops_table`, taking one image patch, one label patch, and
the region's `Roi`. It can return a `DataFrame` or a plain dict of columns; either
way the rows must carry the object id in a `label` column.

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:extract_features"
```

## Step 2: open the OME-Zarr container

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:open_container"
```

## Step 3: set up the inputs

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:setup_transform"
```

## Step 4: use the FeatureExtractorIterator to create a feature table

`measure` runs the measurement over every region and joins the results
into a `FeatureTable` referencing the input label — one call, one table. The
per-region measurements schedule exactly like `reduce`, so a
`mapper=ThreadedMapper("auto")` parallelizes them; the join still happens once,
at the end. The iterator writes nothing: storing the table is your explicit
`add_table` call, where the name, backend and overwrite policy belong.

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:extract"
```

For flows the default join does not fit — a different table type, filtering, or
aggregation — either declare a custom join with `with_join(...)`, or drop down to the loop that
`measure` replaces:

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:manual_extract"
```

## Measuring with a halo

Tiling a large image (`by_grid`, `by_blocks`) cuts objects at the tile
edges — a border nucleus would be measured on half its pixels. `with_halo`
fixes that by reading a margin of context around each tile: the function
receives the grown patches (and the grown `roi`), so a border object is seen
whole by at least one tile. The price is that *every* tile that sees it
measures it, so the same `label` appears more than once. Deduplicating is
your declared join's job, and every normalized row carries two provenance
columns for exactly that: `roi_index` (the region's global index) and
`roi_name`. The default join keeps the duplicates as-is; a declared join
picks one row per object — here the one from the tile that saw the largest
piece. (`roi_index`, `roi_name`, and `_ngio_index` are reserved: a
measurement function returning one of them is refused.)

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:halo_dedup"
```

### Sanity check: read the table back

```python exec="true" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:table_helpers"
```

```python exec="true" html="1" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:read_table_back"
```

## Plot the features

The table is made for exactly this — one column against another (the area converted
to µm²), every dot one nucleus:

```python exec="true" html="1" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:plot_features"
```

Most nuclei sit in one cloud; the small, bright ones in the upper left — condensed
chromatin, dividing or dying — are the kind of subpopulation you measure to find.

## Next steps

- [Iterators](../getting_started/6_iterators.md) — halos, joins, and the read-only
  iterators in full.
- [Distributed processing](distributed_processing.md) — the same measurement split
  across jobs.
- [HCS exploration](hcs_exploration.md) — aggregate feature tables across a plate.
- [Table specifications](../table_specs/overview.md) — how feature tables are stored.
