---
description: Read and write image data object-by-object using a segmentation as a mask.
---

# 4. Masked Images and Labels

Masked images (or labels) are images that are masked by an instance segmentation mask.

In this section we will show how to create a `MaskedImage` object and how to use it to get the data of the image.

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:setup"
```

Similar to the `Image` and `Label` objects, the `MaskedImage` can be initialized from an `OME-Zarr Container` object using the `get_masked_image` method.

Let's create a masked image from the `nuclei` label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_masked_image"
```

Since the `MaskedImage` is a subclass of `Image`, we can use all the methods available for `Image` objects.

The two most notable exceptions are the `get_roi_as_numpy` (or `get_roi_as_dask`) and `set_roi` which now instead of requiring a `roi` object, require an integer `label`.

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:masked_roi_numpy"
```

```python exec="true" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_helpers"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_masked_roi"
```

Additionally we can use the `zoom_factor` argument to get more context around the ROI.
For example we can zoom out the ROI by a factor of `2`:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:masked_roi_zoom"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_masked_roi_zoom"
```

## Masked operations

In addition to the `get_roi_as_numpy` method, the `MaskedImage` class also provides a masked operation method that allows you to perform reading and writing only on the masked pixels.

For these operations we can use the `get_roi_masked` and `set_roi_masked` methods.
For example, we can use the `get_roi_masked` method to get the masked data for a specific label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_roi_masked"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_get_roi_masked"
```

We can also use the `set_roi_masked` method to set the masked data for a specific label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:set_roi_masked"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_after_set_roi_masked"
```

## Masked Labels

The `MaskedLabel` class is a subclass of [`Label`][ngio.Label] and provides the same functionality as the `MaskedImage` class.

The `MaskedLabel` class can be used to create a masked label from an `OME-Zarr Container` object using the `get_masked_label` method.

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_masked_label"
```

## Next steps

- [HCS Plates](5_hcs.md) — scale up from a single image to a whole plate.
- [Iterators](6_iterators.md) — process every object or region without writing the loop yourself.
