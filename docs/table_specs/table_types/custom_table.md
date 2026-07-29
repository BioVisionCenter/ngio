---
description: How custom table types fit into the ngio table architecture, and what to use until the extension API is documented.
---

# Add a Custom Table

ngio allows users to define custom tables that can be used to store any kind of tabular
data. Custom tables are flexible and can be used to represent any kind of data that does
not fit into the predefined table types.

!!! note "Extension API not yet documented"

    The mechanism for registering a custom table type is public but not yet documented in
    full. Until it is, [Generic Tables](generic_table.md) are the supported way to store
    arbitrary tabular data — they accept any pandas `DataFrame` or `AnnData` object and
    round-trip through every [backend](../backend.md).

    If you need a genuinely new table *type* — with its own validation and metadata — open
    an issue on [GitHub](https://github.com/BioVisionCenter/ngio/issues) describing your
    use case.

## See also

- [Tables Overview](../overview.md) — the three-component table architecture.
- [Generic Tables](generic_table.md) — the untyped fallback.
- [Tables API reference](../../api/tables.md) — `TablesContainer` and the table classes.
