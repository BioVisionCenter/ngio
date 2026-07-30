---
description: "Condition table: a flexible table type for experimental conditions and metadata."
---

# Condition table

A condition table represents experimental conditions or metadata associated with images or experiments. It is a flexible table type, so it can hold any kind of metadata about them.

Example condition table:

| Cell type | Drug     | Dose |
|-----------|-----------|------|
| A         | Drug A   | 10   |
| A         | Drug B   | 20   |

## Specifications

### V1

A condition table must include the following metadata fields in the group attributes:

```json5
{
    // Condition table metadata
    "type": "condition_table",
    "table_version": "1",
    // Backend metadata
    "backend": "csv", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "condition_id", // Optional. The column used as the row index.
    "index_type": "str" // Optional. Either "int" or "str"
}
```

As with [generic tables](generic_table.md), a condition table has no default index:
`index_key` and `index_type` appear only when you set one.

In ngio this table type is the [`ConditionTable`][ngio.tables.ConditionTable] class.
