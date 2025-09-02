# 2. Images and Labels

## Images

In order to start working with the image data, we need to instantiate an `Image` object.
ngio provides a high-level API to access the image data at different resolution levels and pixel sizes.

### Getting an image

=== "Highest Resolution Image"
    By default, the `get_image` method returns the highest resolution image:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.get_image() # Get the highest resolution image
    >>> print(ome_zarr_container.get_image()) # markdown-exec: hide
    ```

=== "Specific Pyramid Level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.get_image(path="1") # Get a specific pyramid level
    >>> print(ome_zarr_container.get_image(path="1")) # markdown-exec: hide
    ```
    This will return the image at the specified pyramid level.

=== "Specific Resolution"
    If you want to get an image with a specific pixel size, you can use the `pixel_size` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio import PixelSize
    >>> pixel_size = PixelSize(x=0.65, y=0.65, z=1.0)
    >>> ome_zarr_container.get_image(pixel_size=pixel_size)
    >>> image = ome_zarr_container.get_image(pixel_size=pixel_size) # markdown-exec: hide
    >>> print(image) # markdown-exec: hide
    ```

=== "Nearest Resolution"
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio import PixelSize
    >>> pixel_size = PixelSize(x=0.60, y=0.60, z=1.0)
    >>> ome_zarr_container.get_image(pixel_size=pixel_size, strict=False)
    >>> image = ome_zarr_container.get_image(pixel_size=pixel_size, strict=False) # markdown-exec: hide
    >>> print(image) # markdown-exec: hide
    ```
    This will return the image with the nearest resolution to the requested pixel size.

Similarly to the `OME-Zarr Container`, the `Image` object provides a high-level API to access the image metadata.

=== "Dimensions"
    ```pycon exec="true" source="console" session="get_started"
    >>> image.dimensions
    >>> print(image.dimensions) # markdown-exec: hide
    ```
    The `dimensions` attribute returns a object with the image dimensions for each axis.

=== "Pixel Size"
    ```pycon exec="true" source="console" session="get_started"
    >>> image.pixel_size
    >>> print(image.pixel_size) # markdown-exec: hide
    ```
    The `pixel_size` attribute returns the pixel size for each axis.

=== "On disk array infos"
    ```pycon exec="true" source="console" session="get_started"
    >>> image.shape, image.dtype, image.chunks
    >>> print(image.shape, image.dtype, image.chunks) # markdown-exec: hide
    ```
    The `axes` attribute returns the order of the axes in the image.

### Working with image data

Once you have the `Image` object, you can access the image data as a:

=== "Numpy Array"
    ```pycon exec="true" source="console" session="get_started"
    >>> data = image.get_as_numpy() # Get the image as a numpy array
    >>> data.shape, data.dtype
    >>> print(data.shape, data.dtype) # markdown-exec: hide
    ```

=== "Dask Array"
    ```pycon exec="true" source="console" session="get_started"
    >>> dask_array = image.get_as_dask() # Get the image as a dask array
    >>> dask_array
    >>> print(dask_array) # markdown-exec: hide
    ```

=== "Legacy"
    A generic `get_array` method is still available for backwards compatibility.

    ```pycon exec="true" source="console" session="get_started"
    >>> data = image.get_array(mode="numpy") # Get the image as a numpy or dask or delayed object
    >>> data.shape, data.dtype
    >>> print(data.shape, data.dtype) # markdown-exec: hide
    ```

The `get_as_*` can also be used to slice the image data, and query specific axes in specific orders:

```pycon exec="true" source="console" session="get_started"
>>> image_slice = image.get_as_numpy(c=0, x=slice(0, 128), axes_order=["t", "z", "y", "x", "c"]) # Get a specific channel and axes order
>>> image_slice.shape
>>> print(image_slice.shape) # markdown-exec: hide
```

If you want to edit the image data, you can use the `set_array` method:

```python
>>> image.set_array(data) # Set the image data
```

The `set_array` method can be used to set the image data from a numpy array, dask array, or dask delayed object.

A minimal example of how to use the `get_array` and `set_array` methods:

```python exec="true" source="material-block" session="get_started"
# Get the image data as a numpy array
data = image.get_as_numpy(c=0, x=slice(0, 128), y=slice(0, 128), axes_order=["z", "y", "x", "c"])

# Modify the image data
some_function = lambda x: x # markdown-exec: hide
data = some_function(data)

# Set the modified image data
image.set_array(data, c=0, x=slice(0, 128), y=slice(0, 128), axes_order=["z", "y", "x", "c"])
image.consolidate() # Consolidate the changes to all resolution levels, see below for more details
```

!!! important
    The `set_array` method will overwrite the image data at single resolution level. After you have finished editing the image data, you need to `consolidate` the changes to the OME-Zarr file at all resolution levels:
    ```python
    >>> image.consolidate() # Consolidate the changes
    ```
    This will write the changes to the OME-Zarr file at all resolution levels.

## Labels

`Labels` represent segmentation masks that identify objects in the image. In ngio `Labels` are similar to `Images` and can
be accessed and manipulated in the same way.

### Getting a label

Now let's see what labels are available in our image:

```pycon exec="true" source="console" session="get_started"
# List all available labels
>>> ome_zarr_container.list_labels() # Available labels
>>> print(ome_zarr_container.list_labels()) # markdown-exec: hide
>>> print("") # markdown-exec: hide
```

We have `4` labels available in our image. Let's see how to access them:

=== "Highest Resolution Label"
    By default, the `get_label` method returns the highest resolution label:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.get_label("nuclei") # Get the highest resolution label
    >>> print(ome_zarr_container.get_label("nuclei")) # markdown-exec: hide
    ```

=== "Specific Pyramid Level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.get_label("nuclei", path="1") # Get a specific pyramid level
    >>> print(ome_zarr_container.get_label("nuclei", path="1")) # markdown-exec: hide
    ```
    This will return the label at the specified pyramid level.

=== "Specific Resolution"
    If you want to get a label with a specific pixel size, you can use the `pixel_size` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio import PixelSize
    >>> pixel_size = PixelSize(x=0.65, y=0.65, z=1.0)
    >>> ome_zarr_container.get_label("nuclei", pixel_size=pixel_size)
    >>> label_nuclei = ome_zarr_container.get_label("nuclei", pixel_size=pixel_size) # markdown-exec: hide
    >>> print(label_nuclei) # markdown-exec: hide
    ```

=== "Nearest Resolution"
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio import PixelSize
    >>> pixel_size = PixelSize(x=0.60, y=0.60, z=1.0)
    >>> ome_zarr_container.get_label("nuclei", pixel_size=pixel_size, strict=False)
    >>> label_nuclei = ome_zarr_container.get_label("nuclei", pixel_size=pixel_size, strict=False) # markdown-exec: hide
    >>> print(label_nuclei) # markdown-exec: hide
    ```
    This will return the label with the nearest resolution to the requested pixel size.

### Working with label data

Data access and manipulation for `Labels` is similar to `Images`. You can use the `get_array` and `set_array` methods to access and modify the label data.

### Deriving a label

Often, you might want to create a new label based on an existing image. You can do this using the `derive_label` method:

```pycon exec="true" source="console" session="get_started"
>>> new_label = ome_zarr_container.derive_label("new_label", overwrite=True) # Derive a new label
>>> print(new_label) # markdown-exec: hide
```

This will create a new label with the same dimensions as the original image (without channels) and compatible metadata.
If you want to create a new label with slightly different metadata see [API Reference](../api/images.md).
