---
description: "Feature table: per-object measurements tied to a label image."
---

# Feature table

A feature table is a table type for representing per-object features in an image. Each row in a feature table corresponds to a specific label in the label image.

A feature table can also declare what kind of feature each column holds:

- `measurement`: A quantitative measurement of the object, such as area, perimeter, or intensity.
- `categorical`: A categorical feature of the object, such as a classification label or a type.
- `metadata`: Additional free-form columns that can be used to store any other information about the object, but that should not be used for analysis/classification purposes.

The declaration is there so that downstream tools can select subsets of features without
guessing from dtypes.

!!! warning "Declarative only"

    ngio writes these three lists but does not yet read them back: they do not influence
    how a table is serialised. Casting is decided by dtype alone, as described under
    [table backends](../backend.md). Treat the lists as an annotation for your own
    tooling, not as a contract ngio enforces.

## Specifications

### V1

A feature table must include the following metadata fields in the group attributes:

```json5
{
    // Feature table metadata
    "type": "feature_table",
    "table_version": "1",
    "region": {"path": "../labels/label_DAPI"}, // Path to the label image associated with this feature table
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "label",
    "index_type": "int", // Either "int" or "str"
    "instance_key": "label" // Mirrors index_key; identifies the label each row describes
}
```

ngio also always writes the three feature-type lists, empty when you have not set them:

```json5
{
    "categorical_columns": [
        "label",
        "cell_type"
    ],
    "measurement_columns": [
        "area",
        "perimeter",
        "intensity_mean",
        "intensity_std"
    ],
    "metadata_columns": [
        "description"
    ]
}
```
