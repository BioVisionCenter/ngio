# Table Backends

In ngio we implemented four different table backends. Each table backend is a python class that can serialize tabular data into OME-Zarr containers.

These backends are wrappers around existing tooling implemented in `anndata`, `pandas`, and `polars`.
Currently, we provide a thin layer of metadata and table normalization to ensure that tables are serialized/deserialized in a consistent way across the different backends and across different table objects.

In particular, we provide the metadata that describes the intended index key and type of the table for each backend.

## AnnData Backend

AnnData is a widely used format in single-cell genomics, and can natively store complex tabular data in a Zarr group. The AnnData backend in ngio is a wrapper around the `anndata` library, which performs some table normalization for consistency and compatibility with the ngio table specifications.

The following normalization steps are applied to each table before saving it to the AnnData backend:

- We separate the table in two parts: The floating point columns are casted to `float32` and stored as `X` in the AnnData object, while the categorical, boolean, and integer columns are stored as `obs`.
- The index column is cast to a string, and is stored in the `obs` index.
- The index column name must match the `index_key` specified in the metadata.

AnnData backend metadata:

```json
{
    // Backend metadata
    "backend": "annadata", // the backend used to store the table, e.g. "annadata", "parquet", etc..
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

## Parquet Backend

The Parquet backend is a high-performance columnar storage format that is widely used in big data processing. It is designed to efficiently store large datasets and can be used with various data processing frameworks.
Another advantage of the Parquet backend is that it can be used lazily, meaning that the data is not loaded into memory until it is needed. This can be useful for working with large datasets that do not fit into memory.

Parquet backend metadata:

```json
{
    // Backend metadata
    "backend": "parquet", // the backend used to store the table, e.g. "annadata", "parquet", etc..
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

## CSV Backend

The CSV backend is a plain text format that is widely used for tabular data. It is easy to read and write, and can be used across many different tools.

The CSV backend in ngio follows closely the same specifications as the Parquet backend, with the following metadata:

```json
{
    // Backend metadata
    "backend": "csv", // the backend used to store the table, e.g. "annadata", "parquet", etc..
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

## JSON Backend

The JSON backend serializes the table data into the Zarr group attributes as a JSON object. This backend is useful for tiny tables.

JSON backend metadata:

```json
{
    // Backend metadata
    "backend": "json", // the backend used to store the table, e.g. "annadata", "parquet", etc..
    "index_key": "index", // The default index key for the table, which is used to identify each row.
    "index_type": "int" // Either "int" or "str"
}
```

The table will be stored in a subgroup of the Zarr group, and the metadata will be stored in the group attributes. Storing the table in a subgroup instead of a standalone json file allows for easier access via the Zarr API.

```bash
table.zarr          # Zarr group for the table
└── table           # Zarr subgroup containing the table data
    ├── .zattrs     # the json table data serialized as a JSON object
    └── .zgroup     # Zarr group metadata
├── .zattrs         # Zarr group attributes containing the metadata
└── .zgroup         # Zarr group metadata
```
