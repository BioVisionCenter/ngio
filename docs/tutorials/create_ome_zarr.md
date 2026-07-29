---
description: Convert a numpy array into an OME-Zarr image and attach a ROI table.
---

# OME-Zarr Creation

This is a minimal example of how to create an OME-Zarr image using `ngio`.

This example is just a simple demonstration but for more complex conversion tasks please refer
to the converter tooling library [ome-zarr-converters-tools](https://github.com/BioVisionCenter/ome-zarr-converters-tools).

Let's start by converting a sample image from `skimage` to OME-Zarr format.

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

It is often useful to add ROIs to OME-Zarr images to be able to retrieve them later.
This can be done using the `ngio` library as follows.

```python exec="true" source="material-block" session="create_ome_zarr"
--8<-- "docs/snippets/tutorials/create_ome_zarr.py:add_roi_table"
```

## Next steps

- [Image Processing](image_processing.md) — process the image you just created.
- [OME-Zarr Containers](../getting_started/1_ome_zarr_containers.md) — the container API in depth.
