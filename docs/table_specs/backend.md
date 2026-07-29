---
description: "The on-disk table backends: AnnData, Parquet, CSV and JSON."
---

# Table backends

ngio has four table backends. Each one is a Python class that can serialise tabular data into OME-Zarr containers.

These backends are wrappers around existing tooling implemented in `anndata`, `pandas`, and `polars`.
On top of that, ngio adds a thin layer of metadata and table normalisation, so that tables are serialised and deserialised consistently across the different backends and across different table objects.

In particular, the metadata describes the intended index key and type of the table for each backend.

## AnnData backend

AnnData is a widely used format in single-cell genomics, and can natively store complex tabular data in a Zarr group. The AnnData backend in ngio is a wrapper around the `anndata` library, and applies some table normalisation for consistency and compatibility with the ngio table specifications.

The following normalisation steps are applied to each table before saving it to the AnnData backend:

- The table is separated in two parts: the floating point columns are cast to `float32` and stored as `X` in the AnnData object, while the categorical, boolean, and integer columns are stored as `obs`.
- The index column is cast to a string, and is stored in the `obs` index.
- The index column name must match the `index_key` specified in the metadata.

AnnData backend metadata:

```json
{
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the table, which is used to identify each row.
    "index_type": "str", // Either "int" or "str"
}
```

Additionally, the AnnData package will write some additional metadata to the group attributes

```json
{
    "encoding-type": "anndata",
    "encoding-version": "0.1.0",
}
```

## Parquet backend

The Parquet backend is a high-performance columnar storage format that is widely used in big data processing. It is designed to store large datasets efficiently and can be used with various data processing frameworks.
Another advantage of the Parquet backend is that it can be read lazily: the data is not loaded into memory until it is needed. That helps when working with datasets that do not fit into memory.

Parquet backend metadata:

```json
{
    // Backend metadata
    "backend": "parquet", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the table, which is used to identify each row.
    "index_type": "int", // Either "int" or "str"
}
```

The Zarr group directory will contain the Parquet file, and the metadata will be stored in the group attributes.

```bash
table.zarr          # Zarr group for the table
├── table.parquet   # Parquet file containing the table data
├── .zattrs         # Zarr group attributes containing the metadata
└── .zgroup         # Zarr group metadata
```

## CSV backend

The CSV backend is a plain text format that is widely used for tabular data. It can be read and written by hand, and across many different tools.

The CSV backend in ngio follows closely the same specifications as the Parquet backend, with the following metadata:

```json
{
    // Backend metadata
    "backend": "csv", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the table, which is used to identify each row.
    "index_type": "int", // Either "int" or "str"
}
```

The Zarr group directory will contain the CSV file, and the metadata will be stored in the group attributes.

```bash
table.zarr         # Zarr group for the table
├── table.csv      # CSV file containing the table data
├── .zattrs        # Zarr group attributes containing the metadata
└── .zgroup        # Zarr group metadata
```

## JSON backend

The JSON backend serialises the table data into the Zarr group attributes as a JSON object. This backend is useful for tiny tables.

JSON backend metadata:

```json
{
    // Backend metadata
    "backend": "json", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "index", // The default index key for the table, which is used to identify each row.
    "index_type": "int" // Either "int" or "str"
}
```

The table is stored in a subgroup of the Zarr group, and the metadata is stored in the group attributes. Storing the table in a subgroup rather than a standalone JSON file keeps it accessible through the Zarr API.

```bash
table.zarr          # Zarr group for the table
└── table           # Zarr subgroup containing the table data
    ├── .zattrs     # the table data serialised as a JSON object
    └── .zgroup     # Zarr group metadata
├── .zattrs         # Zarr group attributes containing the metadata
└── .zgroup         # Zarr group metadata
```
