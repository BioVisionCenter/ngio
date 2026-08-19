---
description: Apply a Gaussian blur eagerly, lazily with dask, and through an ngio iterator.
---

# Image processing

**Apply a Gaussian blur three ways.**

Apply a Gaussian blur to an OME-Zarr image three ways with `ngio`: eagerly on a numpy
array, lazily with `dask`, and through an ngio iterator. Along the way you derive a new
container that keeps the metadata of the original and write the processed image into it.

## Step 1: set up

Start with a function that applies a Gaussian blur to an image. It takes an image and a
sigma value as input, and returns the blurred image.

```python exec="true" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:plot_helpers"
```

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:gaussian_blur"
```

## Step 2: open the OME-Zarr container

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:open_container"
```

## Step 3: create a new empty OME-Zarr container

ngio can "derive" a new container from an existing one. Use this when you want to apply
processing to an image and save the results in a new container that preserves the original
metadata and dimensions (unless you change them explicitly when deriving).

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:derive_image"
```

## Step 4: apply the Gaussian blur and consolidate the processed image

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:apply_blur"
```

### Plot the results

Finally, visualise the original and blurred images with `matplotlib`.

```python exec="true" html="1" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:plot_blur"
```

## Step 5: out-of-memory processing

Some images are larger than memory. In that case, use the `dask` library to process the image in chunks: with `ngio` you query the data as a `dask` array and apply the processing function to it.

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:dask_blur"
```

## Step 6: image processing iterators

`ngio` also processes large images with iterators. This API is not meant to replace `dask`: it lets you iterate over arbitrary regions, and it supplies default broadcasting behaviour. Note how it solves the two problems the dask version left open — `with_halo` reads a margin of context so the blur has no seams, and a `ThreadedMapper` fans the regions out on a thread pool while the disjoint write footprints keep the parallel writes safe.

```python exec="true" source="material-block" session="image_processing"
--8<-- "docs/snippets/tutorials/image_processing.py:iterators"
```

## Next steps

- [Image segmentation](image_segmentation.md) — turn images into labels.
- [Iterators](../getting_started/6_iterators.md) — the iterator concepts behind step 6.
