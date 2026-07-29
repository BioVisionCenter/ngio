---
description: Convert a numpy array into an OME-Zarr image and attach a ROI table.
---

# Create an OME-Zarr image

Convert a numpy array into an OME-Zarr image with `ngio`, then attach a ROI table to it.
By the end you will have an on-disk container that the other tutorials read from.

For larger conversion jobs — vendor formats, multi-file acquisitions, whole plates — reach
for the converter tooling library [ome-zarr-converters-tools](https://github.com/BioVisionCenter/ome-zarr-converters-tools).

Start by converting a sample image from `skimage` to OME-Zarr format.

```python exec="true" session="create_ome_zarr"
--8<-- "docs/snippets/tutorials/create_ome_zarr.py:plot_helpers"
```

```python exec="true" html="1" source="material-block" session="create_ome_zarr"
--8<-- "docs/snippets/tutorials/create_ome_zarr.py:plot_input_image"
```

```python exec="true" source="material-block" session="create_ome_zarr"
--8<-- "docs/snippets/tutorials/create_ome_zarr.py:create"
```

## Adding a ROI table to an OME-Zarr image

Attaching ROIs to an OME-Zarr image lets you retrieve those regions later. Add them with
`ngio` as follows.

```python exec="true" source="material-block" session="create_ome_zarr"
--8<-- "docs/snippets/tutorials/create_ome_zarr.py:add_roi_table"
```

## Next steps

- [Image processing](image_processing.md) — process the image you just created.
- [OME-Zarr containers](../getting_started/1_ome_zarr_containers.md) — the container API in depth.
