---
description: Apply a Gaussian blur eagerly, lazily with dask, and through an ngio iterator.
---

# Image Processing

This is a minimal example of how to use the `ngio` library for applying some basic image processing techniques.

For this example we will apply gaussian blur to an image.

## Step 1: Setup

We will first create a simple function to apply gaussian blur to an image. This function will take an image and a sigma value as input and return the blurred image.

```python exec="true" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:plot_helpers"
```

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:gaussian_blur"
```

## Step 2: Open the OME-Zarr container

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:open_container"
```

## Step 3: Create a new empty OME-Zarr container

ngio provides a simple way to "derive" a new container from an existing one. This is useful when you want to apply some processing to an image and save the results in a new container that
preserves the original metadata and dimensions (unless explicitly changed when deriving).

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:derive_image"
```

## Step 4: Apply the gaussian blur and consolidate the processed image

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:apply_blur"
```

### Plot the results

Finally, we can visualize the original and blurred images using `matplotlib`.

```python exec="true" html="1" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:plot_blur"
```

## Step 5: Out of memory processing

Sometimes we want to apply some simple processing to larger than memory images. In this case, we can use the `dask` library to process the image in chunks. In `ngio` we can simply query the data as a `dask` array and apply the desired processing function to it.

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:dask_blur"
```

## Step 6: Image Processing Iterators

`ngio` provides an alternative way to process large images using iterators. This API is not meant to replace `dask` but to provide a simple way to iterate over arbitrary regions, moreover it provides a simple way to implement default broadcasting behaviors.

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:iterators"
```

## Next steps

- [Image Segmentation](image_segmentation.md) — turn images into labels.
- [Iterators](../getting_started/6_iterators.md) — the iterator concepts behind Step 6.
