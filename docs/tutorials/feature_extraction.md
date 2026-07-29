---
description: Extract regionprops features and store them as an ngio feature table.
---

# Feature Extraction

This tutorial covers extracting regionprops features from an image with `ngio` and `skimage`, and writing them back as a table in the OME-Zarr container. Moreover we will also write the features to a table in the ome-zarr container.

## Step 1: Open the OME-Zarr Container

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:extract_features"
```

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:open_container"
```

## Step 2: Setup the inputs

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:setup_transform"
```

## Step 3: Use the FeatureExtractorIterator to create a feature table

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:extract"
```

### Sanity Check: Read the Table back

```python exec="true" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:table_helpers"
```

```python exec="true" html="1" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:read_table_back"
```

## Next steps

- [HCS Exploration](hcs_exploration.md) — aggregate feature tables across a plate.
- [Table Specifications](../table_specs/overview.md) — how feature tables are stored.
