# 5. HCS Plates

Ngio provides a simple interface for high-content screening (HCS) plates. An HCS plate is a collection of OME-Zarr images organized in a grid-like structure. Each plates contains columns and rows, and each well in the plate is identified by its row and column indices. Each well can contain multiple images, and each image can belong to a different acquisition.

The HCS plate is represented by the `OmeZarrPlate` class.

Let's open an `OmeZarrPlate` object.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:setup"
```

This example plate is very small and contains only a single well.

## Plate overview

The `OmeZarrPlate` object provides a high-level overview of the plate, including rows, columns, and acquisitions. The following methods are available:

=== "Columns"
    Show the columns in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_columns"
    ```
=== "Rows"
    Show the rows in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_rows"
    ```
=== "Acquisitions"
    Show the acquisitions ids:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_acquisitions"
    ```

## Retrieving the path to the images

The `OmeZarrPlate` object provides multiple methods to retrieve the path to the images in the plate.

=== "All Images Paths"
    This will return the paths to all images in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:images_paths"
    ```

=== "All Wells Paths"
    This will return the paths to all wells in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:wells_paths"
    ```

=== "All Images Paths in a Well"
    This will return the paths to all images in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:well_images_paths"
    ```

## Getting the images

The `OmeZarrPlate` object provides a method to get the image objects in a well. The method `get_well_images` takes the row and column indices of the well and returns a list of `OmeZarrContainer` objects.

=== "All Images"
    Get all images in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_images"
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "All Images in a Well"
    Get all images in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_well_images"
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "Specific Image"
    Get a specific image in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_image"
    ```
    This will return the `OmeZarrContainer` object for the image in the well.

=== "Filter by Acquisition"
    In these methods, you can also filter the images by acquisition. When available, the `acquisition` parameter can be used to filter the images by acquisition id.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_well_images_by_acquisition"
    ```
    The `acquisition` is not required, and if not provided, an empty dictionary will be returned.

## Creating a plate

Ngio provides a utility function to create a plate.

The first step is to create a list of `ImageInWellPath` objects. Each `ImageInWellPath` object contains the path to the image and the corresponding well.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:image_in_well_paths"
```

!!! note
    The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.

Then, you can create the plate using the `create_empty_plate` function.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:create_empty_plate"
```

This has created a new empty plate with the metadata correctly set. But no images have been added yet.

### Modifying the plate

You can add images or remove images

=== "Add Images"
    To add images to the plate, you can use the `add_image` method. This method takes the row and column indices of the well and the path to the image.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_add_image"
    ```
    This will add a new image to the plate and well metadata.
    !!! note
        The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.
    !!! warning
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_add_image` method instead.

=== "Remove Images"
    To remove images from the plate, you can use the `remove_image` method. This method takes the row and column indices of the well and the path to the image.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_remove_image"
    ```
    This will remove the image metadata from the plate and well metadata.
    !!! warning
        No data will be removed from the store. If an image is saved in the store it will remain there.
        Also the metadata will only be removed from the plate.well metadata. The number of columns and rows will not be updated.
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_remove_image` method instead.
