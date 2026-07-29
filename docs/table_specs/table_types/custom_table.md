---
description: How custom table types fit into the ngio table architecture, and what to use until the extension API is documented.
---

# Add a custom table

A custom table is a table type you define yourself, for data that does not fit any of the
predefined types.

!!! note "Extension API not yet documented"

    The mechanism for registering a custom table type is public but not yet documented in
    full. Until it is, [generic tables](generic_table.md) are the supported way to store
    arbitrary tabular data — they accept any pandas `DataFrame` or `AnnData` object and
    round-trip through every [backend](../backend.md).

    If you need a genuinely new table *type* — with its own validation and metadata — open
    an issue on [GitHub](https://github.com/BioVisionCenter/ngio/issues) describing your
    use case.

## See also

- [Tables overview](../overview.md) — the three-component table architecture.
- [Generic tables](generic_table.md) — the untyped fallback.
- [Tables API reference](../../api/tables.md) — `TablesContainer` and the table classes.
