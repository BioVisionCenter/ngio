# 1. OME-Zarr Container

Let's see how to open and explore an OME-Zarr image using `ngio`:

```python exec="true" source="material-block" session="get_started"
from pathlib import Path
from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data")
download_dir = Path(".").absolute() / "data" # markdown-exec: hide
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr_container = open_ome_zarr_container(image_path)
```

The `OME-Zarr Container` in is your entry point to working with OME-Zarr images. It provides high-level access to the image metadata, images, labels, and tables.

```pycon exec="true" source="console" session="get_started"
>>> ome_zarr_container
>>> print(ome_zarr_container) # markdown-exec: hide
```

The `OME-Zarr Container` will be the starting point for all your image processing tasks.

## Main concepts

### What is the OME-Zarr container?

The `OME-Zarr Container` in ngio is your entry point to working with OME-Zarr images.

It provides:

- **OME-Zarr overview**: get an overview of the OME-Zarr file, including the number of image levels, list of labels, and tables available.
- **Image access**: get access to the images at different resolution levels and pixel sizes.
- **Label management**: check which labels are available, access them, and create new labels.
- **Table management**: check which tables are available, access them, and create new tables.
- **Derive new OME-Zarr images**: create new images based on the original one, with the same or similar metadata.

### What is the OME-Zarr container not?

The `OME-Zarr Container` object does not allow the user to interact with the image data directly. For that, we need to use the `Image`, `Label`, and `Table` objects.

## OME-Zarr overview

Examples of the OME-Zarr metadata access:

=== "Number of Resolution Levels"
    Show the number of resolution levels:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.levels # Show the number of resolution levels
    >>> print(ome_zarr_container.levels) # markdown-exec: hide
    ```

=== "Available Paths"
    Show the paths to all available resolution levels:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.levels_paths # Show the paths to all available images
    >>> print(ome_zarr_container.levels_paths) # markdown-exec: hide
    ```

=== "Dimensionality"
    Show if the image is 2D or 3D:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.is_3d # Get if the image is 3D
    >>> print(ome_zarr_container.is_3d) # markdown-exec: hide
    ```
    or if the image is a time series:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.is_time_series # Get if the image is a time series
    >>> print(ome_zarr_container.is_time_series) # markdown-exec: hide
    ```

=== "Full Metadata Object"
    ```pycon exec="true" source="console" session="get_started"
    >>> metadata = ome_zarr_container.image_meta
    >>> print(metadata) # markdown-exec: hide
    ```
    The metadata object contains all the information about the image, for example, the channel labels:
    ```pycon exec="true" source="console" session="get_started"
    >>> metadata.channels_meta.channel_labels
    >>> print(metadata.channels_meta.channel_labels) # markdown-exec: hide
    ```

## Accessing images / labels / tables

To access images, labels, and tables, you can use the `get_image`, `get_label`, and `get_table` methods of the `OME-Zarr Container` object.

A variety of examples and additional information can be found in the [Images and Labels](./2_images.md), and [Tables](./3_tables.md) sections.

## Creating derived images

When processing an image, you might want to create a new image with the same metadata:

```python
# Create a new image based on the original
new_image = ome_zarr_container.derive_image("data/new_ome.zarr", overwrite=True)
```

This will create a new OME-Zarr image with the same metadata as the original image.
But you can also create a new image with slightly different metadata, for example, with a different shape:

```python
# Create a new image with a different shape
new_image = ome_zarr_container.derive_image(
    "data/new_ome.zarr", 
    overwrite=True, 
    shape=(16, 128, 128), 
    xy_pixelsize=0.65, 
    z_spacing=1.0
)
```

## Creating new images

You can create OME-Zarr images from an existing numpy array using the `create_ome_zarr_from_array` function.

```python
import numpy as np
from ngio import create_ome_zarr_from_array

# Create a random 3D array
x = np.random.randint(0, 255, (16, 128, 128), dtype=np.uint8)

# Save as OME-Zarr
new_ome_zarr_image = create_ome_zarr_from_array(
    store="random_ome.zarr", 
    array=x, 
    xy_pixelsize=0.65, 
    z_spacing=1.0
)
```

Alternatively, if you wanto to create an a empty OME-Zarr image, you can use the `create_empty_ome_zarr` function:

```python
from ngio import create_empty_ome_zarr
# Create an empty OME-Zarr image
new_ome_zarr_image = create_empty_ome_zarr(
    store="empty_ome.zarr", 
    shape=(16, 128, 128), 
    xy_pixelsize=0.65, 
    z_spacing=1.0
)
```

This will create an empty OME-Zarr image with the specified shape and pixel sizes.

## Opening remote OME-Zarr containers

You can use `ngio` to open remote OME-Zarr containers.
For publicly available OME-Zarr containers, you can just use the `open_ome_zarr_container` function with a URL.

For example, to open a remote OME-Zarr container hosted on a github repository:

```python
from ngio.utils import fractal_fsspec_store

url = (
    "https://raw.githubusercontent.com/"
    "fractal-analytics-platform/fractal-ome-zarr-examples/"
    "refs/heads/main/v04/"
    "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
)

store = fractal_fsspec_store(url=url)
ome_zarr_container = open_ome_zarr_container(store)
```

For fractal users, the `fractal_fsspec_store` function can be used to open private OME-Zarr containers.
In this case we need to provide a `fractal_token` to authenticate the user.

```python
from ngio.utils import fractal_fsspec_store
 
store = fractal_fsspec_store(url="https://fractal_url...", fractal_token="**your_secret_token**")
ome_zarr_container = open_ome_zarr_container(store)
```
