---
description: Explore an HCS plate, aggregate tables across images, and create a new plate.
---

# HCS exploration

Open an OME-Zarr plate with `ngio`, see what it contains, aggregate a table across every
image in it, and write the result back to the plate. The last section creates a new empty
plate from scratch.

## Step 1: show what's in the plate

```python exec="true" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:open_plate"
```

## Step 2: aggregate tables across all images

```python exec="true" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:table_helpers"
```

```python exec="true" html="1" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:concatenate_tables"
```

## Step 3: save the table in the plate

```python exec="true" html="1" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:save_table"
```

## Step 4: create a new empty plate

```python exec="true" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:create_plate"
```

## Next steps

- [HCS plates](../getting_started/5_hcs.md) — the plate API in depth.
- [HCS API reference](../api/hcs.md) — `OmeZarrPlate` and `OmeZarrWell`.
