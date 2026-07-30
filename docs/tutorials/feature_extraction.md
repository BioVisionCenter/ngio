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
`skimage.measure.regionprops_table`, taking one image patch and one label patch.

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

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:extract"
```

### Sanity check: read the table back

```python exec="true" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:table_helpers"
```

```python exec="true" html="1" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:read_table_back"
```

## Next steps

- [HCS exploration](hcs_exploration.md) — aggregate feature tables across a plate.
- [Table specifications](../table_specs/overview.md) — how feature tables are stored.
