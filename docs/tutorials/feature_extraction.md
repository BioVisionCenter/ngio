# Feature Extraction

This sections will cover how to extract regionprops features from an image using `ngio`, `skimage`. Moreover we will also write the features to a table in the ome-zarr container.

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

```python exec="true" source="material-block" session="feature_extraction"
--8<-- "docs/snippets/tutorials/feature_extraction.py:read_table_back"
```
