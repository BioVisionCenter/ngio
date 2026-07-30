---
description: How custom table types fit into the ngio table architecture, and what to use until the extension API is documented.
---

# Add a custom table

A custom table is a table type you define yourself, for data that does not fit any of the
predefined types.

!!! note "No public extension API yet"

    The registry that maps a `type` string to a table class is internal, so there is no
    supported way to register your own table type today. Until there is,
    [generic tables](generic_table.md) are the way to store arbitrary tabular data: they
    accept a pandas `DataFrame`, a polars `DataFrame` or `LazyFrame`, or an `AnnData`
    object.

    One caveat when choosing a [backend](../backend.md): only the AnnData backend can
    write an `AnnData` payload. If your generic table holds one, saving it through the
    Parquet, CSV or JSON backend raises `NotImplementedError` — convert it to a
    `DataFrame` first, or keep it on the AnnData backend.

    If you need a genuinely new table *type* — with its own validation and metadata — open
    an issue on [GitHub](https://github.com/BioVisionCenter/ngio/issues) describing your
    use case.

## See also

- [Tables overview](../overview.md) — the three-component table architecture.
- [Generic tables](generic_table.md) — the untyped fallback.
- [Tables API reference](../../api/tables.md) — `TablesContainer` and the table classes.
