---
description: Segment an OME-Zarr image per field of view, then repeat within a mask.
---

# Image Segmentation

This is a minimal tutorial on how to use ngio for image segmentation.

## Step 1: Setup

We will first implement a very simple function to segment an image. We will use skimage to do this.

```python exec="true" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_helpers"
```

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:segmentation_fn"
```

## Step 2: Open the OME-Zarr container

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:open_container"
```

## Step 3: Segment the image

For this example, we will not segment the image all at once. Instead we will iterate over the image FOVs and segment them one by one.

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:segment"
```

### Plot the segmentation

```python exec="true" html="1" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_segmentation"
```

## Step 4: Masked image segmentation

In this example we will use a mask to restrict the segmentation to certain areas of the image.
In this case we will create a simple mask for illustration purposes, but in a real case scenario the mask could come
from another segmentation mask.

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:create_mask"
```

```python exec="true" html="1" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_mask"
```

Note that the next step rebinds `image` to the *masked* image, so the plot below shows
the masked image rather than the original one.

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:masked_segment"
```

```python exec="true" html="1" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_masked_segmentation"
```

## Next steps

- [Feature Extraction](feature_extraction.md) — measure the objects you just segmented.
- [Masked Images and Labels](../getting_started/4_masked_images.md) — read data object-by-object.
