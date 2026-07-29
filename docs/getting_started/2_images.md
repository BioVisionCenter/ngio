# 2. Images and Labels

## Images

In order to start working with the image data, we need to instantiate an `Image` object.
ngio provides a high-level API to access the image data at different resolution levels and pixel sizes.

### Getting an image

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_container"
```

=== "Highest Resolution Image"
    By default, the `get_image` method returns the highest resolution image:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_default"
    ```

=== "Specific Pyramid Level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_by_path"
    ```
    This will return the image at the specified pyramid level.

=== "Specific Resolution"
    If you want to get an image with a specific pixel size, you can use the `pixel_size` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_by_pixel_size"
    ```

=== "Nearest Resolution"
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_nearest"
    ```
    This will return the image with the nearest resolution to the requested pixel size.

Similarly to the `OME-Zarr Container`, the `Image` object provides a high-level API to access the image metadata.

=== "Dimensions"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_dimensions"
    ```
    The `dimensions` attribute returns a object with the image dimensions for each axis.

=== "Pixel Size"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_pixel_size"
    ```
    The `pixel_size` attribute returns the pixel size for each axis.

=== "On disk array infos"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_array_info"
    ```
    The `axes` attribute returns the order of the axes in the image.

### Working with image data

Once you have the `Image` object, you can access the image data as a:

=== "Numpy Array"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_as_numpy"
    ```

=== "Dask Array"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_as_dask"
    ```

=== "Legacy"
    A generic `get_array` method is still available for backwards compatibility.

    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_get_array_legacy"
    ```

The `get_as_*` can also be used to slice the image data, and query specific axes in specific orders:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:image_slice"
```

If you want to edit the image data, you can use the `set_array` method:

```python
>>> image.set_array(data) # Set the image data
```

The `set_array` method can be used to set the image data from a numpy array, dask array, or dask delayed object.

A minimal example of how to use the `get_array` and `set_array` methods:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:set_array_example"
```

!!! important
    The `set_array` method will overwrite the image data at single resolution level. After you have finished editing the image data, you need to `consolidate` the changes to the OME-Zarr file at all resolution levels:
    ```python
    >>> image.consolidate() # Consolidate the changes
    ```
    This will write the changes to the OME-Zarr file at all resolution levels.

### World coordinates slicing

To read or write a specific region of the image defined in world coordinates, you can use the `Roi` object.

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:roi_slicing"
```

## Labels

`Labels` represent segmentation masks that identify objects in the image. In ngio `Labels` are similar to `Images` and can
be accessed and manipulated in the same way.

### Getting a label

Now let's see what labels are available in our image:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:list_labels"
```

We have `4` labels available in our image. Let's see how to access them:

=== "Highest Resolution Label"
    By default, the `get_label` method returns the highest resolution label:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_default"
    ```

=== "Specific Pyramid Level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_by_path"
    ```
    This will return the label at the specified pyramid level.

=== "Specific Resolution"
    If you want to get a label with a specific pixel size, you can use the `pixel_size` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_by_pixel_size"
    ```

=== "Nearest Resolution"
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_nearest"
    ```
    This will return the label with the nearest resolution to the requested pixel size.

### Working with label data

Data access and manipulation for `Labels` is similar to `Images`. You can use the `get_array` and `set_array` methods to access and modify the label data.

### Deriving a label

Often, you might want to create a new label based on an existing image. You can do this using the `derive_label` method:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:derive_label"
```

This will create a new label with the same dimensions as the original image (without channels) and compatible metadata.
If you want to create a new label with slightly different metadata see [API Reference](../api/images.md).
