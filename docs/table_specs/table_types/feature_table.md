# Feature Tables

A feature table is a table type for representing per object features in an image. Each row in a feature table corresponds to a specific label in the label image.

Feature tables can optionally include metadata to specify the type of features stored in each column:

- `measurement`: A quantitative measurement of the object, such as area, perimeter, or intensity.
- `categorical`: A categorical feature of the object, such as a classification label or a type.
- `metadata`: Additional free-from columns that can be used to store any other information about the object, but that should not be used for analysis/classification purposes.

These feature types inform casting of the values when serialising a table and can be used in downstream analysis to select specific subsets of features. The feature type can be explicitly specified in the feature table metadata. Alternatively, if a column is not specified, we apply the following casting rules:

- If the column contains only numeric values, it is considered a `measurement`.
- If the column contains string or boolean values, it is considered a `categorical`.
- The index column is considered a `categorical` feature.

## Specifications

### V1

A feature table must include the following metadata fields in the group attributes:

```json
{
    // Feature table metadata
    "type": "feature_table",
    "table_version": "1",
    "region": {"path": "../labels/label_DAPI"}, // Path to the label image associated with this feature table
    // Backend metadata
    "backend": "annadata", // the backend used to store the table, e.g. "annadata", "parquet", etc..
    "index_key": "label",
    "index_type": "int", // Either "int" or "str"
}
```

Additionally, it can include feature type information such as:

```json
{
    "categorical_columns": [
        "label",
        "cell_type",
    ],
    "measurement_columns": [
        "area",
        "perimeter",
        "intensity_mean",
        "intensity_std"
    ],
    "metadata_columns": [
        "description",
    ],
}
```
