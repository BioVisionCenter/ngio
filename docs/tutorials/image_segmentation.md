---
description: Segment an OME-Zarr image per field of view, then repeat within a mask.
---

# Image segmentation

Segment an OME-Zarr image with `ngio` and `skimage`, one field of view at a time, and
write the result back as a label. The second half repeats the segmentation inside a mask,
so it only runs where you want it to.

## Step 1: set up

Start with a function that segments an image, using `skimage` to do the work.

```python exec="true" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_helpers"
```

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:segmentation_fn"
```

## Step 2: open the OME-Zarr container

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:open_container"
```

## Step 3: segment the image

Rather than segmenting the image all at once, iterate over its FOVs and segment them one by one.

```python exec="true" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:segment"
```

### Plot the segmentation

```python exec="true" html="1" source="material-block" session="image_segmentation"
--8<-- "docs/snippets/tutorials/image_segmentation.py:plot_segmentation"
```

## Step 4: masked image segmentation

Now use a mask to restrict the segmentation to certain areas of the image. Here you create
the mask by hand for illustration, but in a real pipeline it would usually come from
another segmentation.

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

- [Feature extraction](feature_extraction.md) — measure the objects you just segmented.
- [Masked images and labels](../getting_started/4_masked_images.md) — read data object-by-object.

## Beyond the tutorials

The [ngio workshop](https://github.com/BioVisionCenter/ngio-workshop) has hands-on marimo
notebooks covering containers, images, labels and tables, and the processing iterators. Run
them locally with `uv`, in the browser via molab, or read them as
[static pages](https://biovisioncenter.github.io/ngio-workshop/).
