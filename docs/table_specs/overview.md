---
description: "The ngio table architecture: backends, in-memory objects and type specifications."
---

# Tables overview

ngio's architecture tightly integrates image and tabular data. To do that, ngio defines custom specifications for serialising and deserialising tabular data into OME-Zarr containers, together with semantically typed tables derived from the [fractal table specification](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/).

## Architecture

The ngio tables architecture is composed of three main components:

### 1. Table backends

A backend module is a class that can serialise tabular data into OME-Zarr containers. ngio supports four on-disk file formats:

- **AnnData**: Commonly used in single-cell genomics, and the standard table for the initial Fractal table spec.
- **Parquet**: A columnar storage file format optimised for large datasets.
- **CSV**: A plain text format for tabular data, readable and writable by hand.
- **JSON**: A lightweight data interchange format that is both readable and efficient for small tables.

For a detailed description of the backend module, see the [table backends documentation](backend.md).

### 2. In-memory table objects

These are Python objects that represent the tabular data in memory. They give you an interface for manipulating and analysing the data without interacting directly with the underlying file format. ngio supports the following in-memory table objects:

- **Pandas DataFrame**: The most commonly used data structure for tabular data in Python.
- **Polars LazyFrame**: A fast DataFrame implementation that allows for lazy evaluation and efficient computation on large datasets.
- **AnnData**: A specialised data structure for single-cell genomics data, which goes beyond plain tabular data.

ngio also provides utilities to convert between these in-memory representations in a standardised way, based on the table type specifications and metadata.

### 3. Table type specifications

These specifications define structured tables that standardise common table types used in image analysis. ngio defines five table types so far:

- **Generic tables**: A flexible table type that can represent any tabular data. See more in the [generic tables documentation](table_types/generic_table.md).
- **ROI tables**: A table type designed for representing Regions of Interest (ROIs) in images. See more in the [ROI tables documentation](table_types/roi_table.md).
- **Masking ROI tables**: A specialised table type for representing ROIs that are associated with specific labels in an OME-Zarr label image. See more in the [masking ROI tables documentation](table_types/masking_roi_table.md).
- **Feature tables**: A table type for representing features extracted from images. This table is also associated with a specific label image. See more in the [feature tables documentation](table_types/feature_table.md).
- **Condition tables**: A table to represent experimental conditions or metadata associated with images or experiments. See more in the [condition tables documentation](table_types/condition_table.md).

Of these, four are recognised automatically when a table is read: ROI, masking ROI, feature
and condition. Anything else — including a table written by another tool — is loaded as a
generic table.

There is also [`GenericRoiTable`][ngio.tables.GenericRoiTable], a ROI table without the
naming and indexing conventions of the standard one. It has no spec page, and is not
auto-detected on read: reach it explicitly with `get_generic_roi_table`.

## Table groups

Tables in OME-Zarr images are organised into groups of tables. Each group is saved in a Zarr group, and can be associated with a specific image or plate. The table groups are:

- **Image tables**: These tables are a sub group of the OME-Zarr image group and contain metadata or features related only to that specific image. The `.zarr` hierarchy is based on image [specification in NGFF 0.4](https://ngff.openmicroscopy.org/0.4/index.html#image-layout). The subgroup structure is based on the approach of the OME-Zarr `labels` group.

```bash
image.zarr        # Zarr group for a OME-Zarr image
|
├── 0             # Zarr array for multiscale level 0
├── ...
├── N             # Zarr array for multiscale level N
|
├── labels        # Zarr subgroup with a list of labels associated to this image
|   ├── label_A   # Zarr subgroup for a given label
|   ├── label_B   # Zarr subgroup for a given label
|   └── ...
|
└── tables        # Zarr subgroup with a list of tables associated to this image
    ├── table_1   # Zarr subgroup for a given table
    ├── table_2   # Zarr subgroup for a given table
    └── ...
```

- **Plate tables**: These tables are a sub group of the OME-Zarr plate group and contain metadata or features related only to that specific plate.

```bash
plate.zarr       # Zarr group for a OME-Zarr HCS plate
|
├── A             # Row A of the plate
|   ├── 1         # Column 1 of row A, i.e. well A1
|   |   ├── 0     # Image 0 in well A1
|   |   ├── 1     # Image 1 in well A1
|   |   └── ...   # Other images in well A1, one per field and acquisition
...
├── tables        # Zarr subgroup with a list of tables associated to this plate
|   ├── table_1   # Zarr subgroup for a given table
|   ├── table_2   # Zarr subgroup for a given table
|   └── ...
└── ...
```

If a plate table contains per-image information, the table should contain `row`, `column`, and `path_in_well` columns.

## Table group attributes

The Zarr attributes of the tables group must include the key tables, pointing to the list of all tables. This makes it easier to discover the tables associated with the current OME-Zarr image or plate.

```json
{
    "tables": ["table_1", "table_2"]
}
```
