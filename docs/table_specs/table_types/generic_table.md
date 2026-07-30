---
description: "Generic table: the untyped table for arbitrary tabular data."
---

# Generic table

A generic table is a flexible table type that can represent any tabular data. It is not tied to any specific domain or use case, which makes it suitable for a wide range of custom applications.

Generic tables are also the safe fallback when you read a table that does not match any other table type.

## Specifications

### V1

A generic table should include the following metadata fields in the group attributes:

```json5
{
    // Generic table metadata
    "type": "generic_table",
    "table_version": "1",
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "my_id", // Optional. The column used as the row index.
    "index_type": "int" // Optional. Either "int" or "str"
}
```

Generic tables have no default index. `index_key` and `index_type` are written only when
you set one, so a generic table saved without an index key carries just `type`,
`table_version` and `backend`.
