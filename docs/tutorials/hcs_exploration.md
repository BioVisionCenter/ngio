# HCS Plates

This is a minimal example of how to work with OME-Zarr Plates using `ngio`.

## Show what's in the plate

```python exec="true" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:open_plate"
```

## Aggregate tables across all images

```python exec="true" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:table_helpers"
```

```python exec="true" html="1" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:concatenate_tables"
```

## Save the table in the HCS plate

```python exec="true" html="1" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:save_table"
```

## Create a new empty Plate

```python exec="true" source="material-block" session="hcs_exploration"
--8<-- "docs/snippets/tutorials/hcs_exploration.py:create_plate"
```
