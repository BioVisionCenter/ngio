---
description: "Load and create ngio tables: ROI, masking ROI, feature and generic tables."
---

# 3. Tables

**Keep ROIs, features and measurements alongside the image.**

Tables are not part of the core OME-Zarr specification, but ngio uses them to store regions
of interest (ROIs), per-object measurements and other tabular data next to the pixel data.
The on-disk layout follows [Fractal's table spec](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/).

## Getting a table

List all available tables and load a specific one:

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_container"
```

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_image"
```

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:list_tables"
```

ngio recognises four typed tables — `roi_table`, `masking_roi_table`, `feature_table` and `condition_table` — plus the untyped `generic_table`, which is what anything it cannot classify is loaded as. The three you will meet most often are below; see the [table specifications](../table_specs/overview.md) for the rest.

=== "ROI table"
    ROI tables can be used to store arbitrary regions of interest (ROIs) in the image.
    For example, load the `FOV_ROI_table`, which contains the microscope field of view (FOV) ROIs:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:roi_table_get"
    ```
    ```python exec="true" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_helpers"
    ```
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_fov_roi_on_image"
    ```
    `get` returns the single ROI with that name; `rois()` returns them all as a list.
    A ROI can then be used to slice the image data:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:roi_table_slice_image"
    ```
    This will return the image data for the specified ROI.
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_fov_roi_crop"
    ```

=== "Masking ROI table"
    Masking ROIs are a special type of ROIs that can be used to store ROIs for masked objects in the image.
    The `nuclei_ROI_table` contains the masks for the `nuclei` label in the image, and is indexed by the label id.
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:masking_table_get"
    ```
    ROIs can be used to slice the image data:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:masking_table_slice_image"
    ```
    This will return the image data for the specified ROI.
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_masking_roi_crop"
    ```
    See [4. Masked images and labels](./4_masked_images.md) for more details on how to use the masking ROIs to load masked data.

=== "Feature table"
    Feature tables are used to store measurements and are indexed by the label id
    ```python exec="true" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:table_helpers"
    ```
    ```python exec="true" html="1" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:feature_table"
    ```

## Creating a table

Tables (unlike images and labels) can be purely in-memory objects, and don't need to be saved on disk.

=== "Creating a ROI table"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_roi_table"
    ```
    If you would like to create on-the-fly a ROI table for the whole image:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:build_image_roi_table"
    ```
    The `build_image_roi_table` method will create a ROI table with a single ROI that covers the whole image.
    This table is not associated with the image and is purely in memory.
    To save it to disk, use the `add_table` method:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:add_roi_table"
    ```

=== "Creating a masking ROI table"
    As with the ROI table, you can create a masking ROI table on the fly, here for the `nuclei` label:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:build_masking_roi_table"
    ```

=== "Creating a feature table"
    Feature tables can be created from a pandas `Dataframe`:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_feature_table"
    ```

=== "Creating a generic table"
    Sometimes you might want to create a table that doesn't fit into the `ROI`, `Masking ROI`, or `Feature` categories.
    In this case, you can use the [`GenericTable`][ngio.tables.GenericTable] class, which allows you to store any tabular data.
    It can be created from a pandas `Dataframe`:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_generic_table"
    ```
    Or from an `AnnData` object:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:generic_table_from_anndata"
    ```

## Next steps

- [Masked images and labels](4_masked_images.md) — use masking ROI tables to read per-object data.
- [Table specifications](../table_specs/overview.md) — the on-disk format behind these tables.
- [Tables API reference](../api/tables.md) — the table classes and backends.
