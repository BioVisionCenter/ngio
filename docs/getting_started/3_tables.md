# 3. Tables

Tables are not part of the core OME-Zarr specification but can be used in ngio to store measurements, features, regions of interest (ROIs), and other tabular data. Ngio follows the [Fractal's Table Spec](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/).

## Getting a table

We can list all available tables and load a specific table:

```pycon exec="true" source="console" session="get_started"
# List all available tables
>>> ome_zarr_container.list_tables()
>>> list_tables = ome_zarr_container.list_tables() # markdown-exec: hide
>>> print(list_tables) # markdown-exec: hide
```

Ngio supports three types of tables: `roi_table`, `feature_table`, and `masking_roi_table`, as well as untyped `generic_table`.

=== "ROI Table"
    ROI tables can be used to store arbitrary regions of interest (ROIs) in the image.
    Here for example we will load the `FOV_ROI_table` that contains the microscope field of view (FOV) ROIs:
    ```pycon exec="true" source="console" session="get_started"
    >>> roi_table = ome_zarr_container.get_table("FOV_ROI_table") # Get a ROI table
    >>> roi_table.get("FOV_1")
    >>> print(roi_table.get("FOV_1")) # markdown-exec: hide
    ```
    ```python exec="1" html="1" session="get_started"
    from io import StringIO
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a random colormap for labels
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    np.random.seed(0)
    cmap_array = np.random.rand(1000, 3)
    cmap_array[0] = 0
    cmap = ListedColormap(cmap_array)
    image_3 = ome_zarr_container.get_image(path="3")
    image_data = image_3.get_array(c=0)
    image_data = np.squeeze(image_data)
    roi = roi_table.get("FOV_1")
    roi = roi.to_roi_pixels(pixel_size=image_3.pixel_size, dimensions=image_3.dimensions)
    #label_3 = ome_zarr_container.get_label("nuclei", pixel_size=image_3.pixel_size)
    #label_data = label_3.get_array()
    #label_data = np.squeeze(label_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("FOV_1 ROI")
    ax.imshow(image_data, cmap='gray')
    ax.add_patch(Rectangle((roi.x, roi.y), roi.x_length, roi.y_length, edgecolor='red', facecolor='none', lw=2))
    #ax.imshow(label_data, cmap=cmap, alpha=0.6)
    # make sure the roi is centered
    ax.axis('off')
    fig.tight_layout()
    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    print(buffer.getvalue())
    ```
    This will return all the ROIs in the table.
    ROIs can be used to slice the image data:
    ```pycon exec="true" source="console" session="get_started"
    >>> roi = roi_table.get("FOV_1")
    >>> roi_data = image.get_roi(roi)
    >>> roi_data.shape
    >>> print(roi_data.shape) # markdown-exec: hide
    ```
    This will return the image data for the specified ROI.
    ```python exec="1" html="1" session="get_started"
    from io import StringIO
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a random colormap for labels
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    np.random.seed(0)
    cmap_array = np.random.rand(1000, 3)
    cmap_array[0] = 0
    cmap = ListedColormap(cmap_array)
    roi = roi_table.get("FOV_1")
    image_3 = ome_zarr_container.get_image(path="3")
    image_data = image_3.get_roi(roi, c=0)
    image_data = np.squeeze(image_data)
    #label_3 = ome_zarr_container.get_label("nuclei", pixel_size=image_3.pixel_size)
    #label_data = label_3.get_array()
    #label_data = np.squeeze(label_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("FOV_1 ROI")
    ax.imshow(image_data, cmap='gray')
    #ax.imshow(label_data, cmap=cmap, alpha=0.6)
    # make sure the roi is centered
    ax.axis('off')
    fig.tight_layout()
    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    print(buffer.getvalue())
    ```

=== "Masking ROI Table"
    Masking ROIs are a special type of ROIs that can be used to store ROIs for masked objects in the image.
    The `nuclei_ROI_table` contains the masks for the `nuclei` label in the image, and is indexed by the label id.
    ```pycon exec="true" source="console" session="get_started"
    >>> masking_table = ome_zarr_container.get_table("nuclei_ROI_table") # Get a mask table
    >>> masking_table.get_label(1)
    >>> print(masking_table.get_label(100)) # markdown-exec: hide
    ```
    ROIs can be used to slice the image data:
    ```pycon exec="true" source="console" session="get_started"
    >>> roi = masking_table.get_label(100)
    >>> roi_data = image.get_roi(roi)
    >>> roi_data.shape
    >>> print(roi_data.shape) # markdown-exec: hide
    ```
    This will return the image data for the specified ROI.
    ```python exec="1" html="1" session="get_started"
    from io import StringIO
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a random colormap for labels
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    np.random.seed(0)
    cmap_array = np.random.rand(1000, 3)
    cmap_array[0] = 0
    cmap = ListedColormap(cmap_array)
    roi = masking_table.get_label(100)
    image_3 = ome_zarr_container.get_image(path="2")
    image_data = image_3.get_roi(roi, c=0)
    image_data = np.squeeze(image_data)
    label_3 = ome_zarr_container.get_label("nuclei", pixel_size=image_3.pixel_size)
    label_data = label_3.get_roi(roi)
    label_data = np.squeeze(label_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Label 1 ROI")
    ax.imshow(image_data, cmap='gray')
    ax.imshow(label_data, cmap=cmap, alpha=0.6)
    # make sure the roi is centered
    ax.axis('off')
    fig.tight_layout()
    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    print(buffer.getvalue())
    ```
    See [4. Masked Images and Labels](./4_masked_images.md) for more details on how to use the masking ROIs to load masked data.

=== "Features Table"
    Features tables are used to store measurements and are indexed by the label id
    ```pycon exec="true" source="material-block" session="get_started"
    >>> feature_table = ome_zarr_container.get_table("regionprops_DAPI") # Get a feature table
    >>> feature_table.dataframe.head(5) # only show the first 5 rows
    >>> print(feature_table.dataframe.head(5).to_markdown()) # markdown-exec: hide
    ```

## Creating a table

Tables (differently from Images and Labels) can be purely in memory objects, and don't need to be saved on disk.

=== "Creating a ROI Table"
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio.tables import RoiTable
    >>> from ngio import Roi
    >>> roi = Roi(x=0, y=0, x_length=128, y_length=128, name="FOV_1")
    >>> roi_table = RoiTable(rois=[roi])
    >>> print(roi_table) # markdown-exec: hide
    ```
    If you would like to create on-the-fly a ROI table for the whole image:
    ```pycon exec="true" source="console" session="get_started"
    >>> roi_table = ome_zarr_container.build_image_roi_table("whole_image")
    >>> roi_table
    >>> print(ome_zarr_container.build_image_roi_table("whole_image")) # markdown-exec: hide
    ```
    The `build_image_roi_table` method will create a ROI table with a single ROI that covers the whole image.
    This table is not associated with the image and is purely in memory.
    If we want to save it to disk, we can use the `add_table` method:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.add_table("new_roi_table", roi_table, overwrite=True)
    >>> roi_table = ome_zarr_container.get_table("new_roi_table")
    >>> print(roi_table) # markdown-exec: hide
    ```

=== "Creating a Masking ROI Table"
    Similarly to the ROI table, we can create a masking ROI table on-the-fly:
    Let's for example create a masking ROI table for the `nuclei` label:
    ```pycon exec="true" source="console" session="get_started"
    >>> masking_table = ome_zarr_container.build_masking_roi_table("nuclei")
    >>> masking_table
    >>> print(ome_zarr_container.build_masking_roi_table("nuclei")) # markdown-exec: hide
    ```

=== "Creating a Feature Table"
    Feature tables can be created from a pandas `Dataframe`:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio.tables import FeatureTable
    >>> import pandas as pd
    >>> example_data = pd.DataFrame({"label": [1, 2, 3], "area": [100, 200, 300]})
    >>> feature_table = FeatureTable(table_data=example_data)
    >>> feature_table
    >>> print(feature_table) # markdown-exec: hide
    ```

=== "Creating a Generic Table"
    Sometimes you might want to create a table that doesn't fit into the `ROI`, `Masking ROI`, or `Feature` categories.
    In this case, you can use the `GenericTable` class, which allows you to store any tabular data.
    It can be created from a pandas `Dataframe`:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio.tables import GenericTable
    >>> import pandas as pd
    >>> example_data = pd.DataFrame({"area": [100, 200, 300], "perimeter": [50, 60, 70]})
    >>> generic_table = GenericTable(table_data=example_data)
    >>> generic_table
    >>> print(generic_table) # markdown-exec: hide
    ```
    Or from an "AnnData" object:
    ```pycon exec="true" source="console" session="get_started"
    >>> from ngio.tables import GenericTable
    >>> import anndata as ad
    >>> adata = ad.AnnData(X=np.random.rand(10, 5), obs=pd.DataFrame({"cell_type": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]}))
    >>> generic_table = GenericTable(table_data=adata)
    >>> generic_table
    >>> print(generic_table) # markdown-exec: hide
    ```
    The `GenericTable` class allows you to store any tabular data, and is a flexible way to work with tables in ngio.
