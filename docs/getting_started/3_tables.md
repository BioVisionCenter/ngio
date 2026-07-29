# 3. Tables

Tables are not part of the core OME-Zarr specification but can be used in ngio to store measurements, features, regions of interest (ROIs), and other tabular data. Ngio follows the [Fractal's Table Spec](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/).

## Getting a table

We can list all available tables and load a specific table:

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_container"
```

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_image"
```

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:list_tables"
```

Ngio supports three types of tables: `roi_table`, `feature_table`, and `masking_roi_table`, as well as untyped `generic_table`.

=== "ROI Table"
    ROI tables can be used to store arbitrary regions of interest (ROIs) in the image.
    Here for example we will load the `FOV_ROI_table` that contains the microscope field of view (FOV) ROIs:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:roi_table_get"
    ```
    ```python exec="true" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_helpers"
    ```
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_fov_roi_on_image"
    ```
    This will return all the ROIs in the table.
    ROIs can be used to slice the image data:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:roi_table_slice_image"
    ```
    This will return the image data for the specified ROI.
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_fov_roi_crop"
    ```

=== "Masking ROI Table"
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
    See [4. Masked Images and Labels](./4_masked_images.md) for more details on how to use the masking ROIs to load masked data.

=== "Features Table"
    Features tables are used to store measurements and are indexed by the label id
    ```python exec="true" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:table_helpers"
    ```
    ```python exec="true" html="1" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:feature_table"
    ```

## Creating a table

Tables (differently from Images and Labels) can be purely in memory objects, and don't need to be saved on disk.

=== "Creating a ROI Table"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_roi_table"
    ```
    If you would like to create on-the-fly a ROI table for the whole image:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:build_image_roi_table"
    ```
    The `build_image_roi_table` method will create a ROI table with a single ROI that covers the whole image.
    This table is not associated with the image and is purely in memory.
    If we want to save it to disk, we can use the `add_table` method:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:add_roi_table"
    ```

=== "Creating a Masking ROI Table"
    Similarly to the ROI table, we can create a masking ROI table on-the-fly:
    Let's for example create a masking ROI table for the `nuclei` label:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:build_masking_roi_table"
    ```

=== "Creating a Feature Table"
    Feature tables can be created from a pandas `Dataframe`:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_feature_table"
    ```

=== "Creating a Generic Table"
    Sometimes you might want to create a table that doesn't fit into the `ROI`, `Masking ROI`, or `Feature` categories.
    In this case, you can use the `GenericTable` class, which allows you to store any tabular data.
    It can be created from a pandas `Dataframe`:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:create_generic_table"
    ```
    Or from an "AnnData" object:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:generic_table_from_anndata"
    ```
    The `GenericTable` class allows you to store any tabular data, and is a flexible way to work with tables in ngio.
