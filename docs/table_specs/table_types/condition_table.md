# Condition Table

A condition table is a simple table that can be used to represent experimental conditions or metadata associated with images or experiments. It is a flexible table type that can be used to store any kind of metadata related to the images or experiments.

Example condition table:

| Cell Type | Drug     | Dose |
|-----------|-----------|------|
| A         | Drug A   | 10   |
| A         | Drug B   | 20   |

## Specifications

### V1

A condition table must include the following metadata fields in the group attributes:

```json
{
    // Condition table metadata
    "type": "condition_table",
    "table_version": "1",
    // Backend metadata
    "backend": "csv", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the condition table, which is used to identify each row.
    "index_type": "int" // Either "int" or "str"
}
```
