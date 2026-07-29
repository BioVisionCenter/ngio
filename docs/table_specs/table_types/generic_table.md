# Generic Tables

A generic table is a flexible table type that can represent any tabular data. It is not tied to any specific domain or use case, making it suitable for a wide range of custom applications.

Generic tables can be used as a safe fallback when trying to read a table that does not match any other specific table type.

## Specifications

### V1

A generic table should include the following metadata fields in the group attributes:

```json
{
    // Generic table metadata
    "type": "generic_table",
    "table_version": "1",
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the generic table, which is used to identify each row.
    "index_type": "int" // Either "int" or "str"
}
```
