# 4. Masked Images and Labels

Masked images (or labels) are images that are masked by an instance segmentation mask.

In this section we will show how to create a `MaskedImage` object and how to use it to get the data of the image.

```python exec="true" session="masked_images"
from pathlib import Path
from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path(".").absolute() / "data" 
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr_container = open_ome_zarr_container(image_path)
```

Similar to the `Image` and `Label` objects, the `MaskedImage` can be initialized from an `OME-Zarr Container` object using the `get_masked_image` method.

Let's create a masked image from the `nuclei` label:

```pycon exec="true" source="console" session="masked_images"
>>> masked_image = ome_zarr_container.get_masked_image("nuclei")
>>> masked_image
>>> print(masked_image) # markdown-exec: hide
```

Since the `MaskedImage` is a subclass of `Image`, we can use all the methods available for `Image` objects.

The two most notable exceptions are the `get_roi_as_numpy` (or `get_roi_as_dask`) and `set_roi` which now instead of requiring a `roi` object, require an integer `label`.

```pycon exec="true" source="console" session="masked_images"
>>> roi_data = masked_image.get_roi_as_numpy(label=1009, c=0)
>>> roi_data.shape
>>> print(roi_data.shape) # markdown-exec: hide
```

```python exec="1" html="1" session="masked_images"
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

image_data = masked_image.get_roi_as_numpy(label=1009, c=0)
image_data = np.squeeze(image_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Label 1009 ROI")
ax.imshow(image_data, cmap='gray')

ax.axis('off')
fig.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

Additionally we can used the `zoom_factor` argument to get more context around the ROI.
For example we can zoom out the ROI by a factor of `2`:

```pycon exec="true" source="console" session="masked_images"
>>> roi_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
>>> roi_data.shape
>>> print(roi_data.shape) # markdown-exec: hide
```

```python exec="1" html="1" session="masked_images"
image_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
image_data = np.squeeze(image_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Label 1009 ROI - Zoomed out")
ax.imshow(image_data, cmap='gray')

ax.axis('off')
fig.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

## Masked operations

In addition to the `get_roi_as_numpy` method, the `MaskedImage` class also provides a masked operation method that allows you to perform reading and writing only on the masked pixels.

For these operations we can use the `get_roi_masked` and `set_roi_masked` methods.
For example, we can use the `get_roi_masked` method to get the masked data for a specific label:

```pycon exec="true" source="console" session="masked_images"
>>> masked_roi_data = masked_image.get_roi_masked(label=1009, c=0, zoom_factor=2)
>>> masked_roi_data.shape
>>> print(masked_roi_data.shape) # markdown-exec: hide
```

```python exec="1" html="1" session="masked_images"
masked_roi_data = masked_image.get_roi_masked(label=1009, c=0, zoom_factor=2)
masked_roi_data = np.squeeze(masked_roi_data)
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Masked Label 1009 ROI")
ax.imshow(masked_roi_data, cmap='gray')
ax.axis('off')
fig.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

We can also use the `set_roi_masked` method to set the masked data for a specific label:

```pycon exec="true" source="console" session="masked_images"
>>> masked_data = masked_image.get_roi_masked(label=1009, c=0)
>>> masked_data = np.random.randint(0, 255, masked_data.shape, dtype=np.uint8)
>>> masked_image.set_roi_masked(label=1009, c=0, patch=masked_data)
```

```python exec="1" html="1" session="masked_images"
masked_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
masked_data = np.squeeze(masked_data)
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Masked Label 1009 ROI - After setting")
ax.imshow(masked_data, cmap='gray')
ax.axis('off')
fig.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

## Masked Labels

The `MaskedLabel` class is a subclass of `Label` and provides the same functionality as the `MaskedImage` class.

The `MaskedLabel` class can be used to create a masked label from an `OME-Zarr Container` object using the `get_masked_label` method.

```pycon exec="true" source="console" session="masked_images"
>>> masked_label = ome_zarr_container.get_masked_label(label_name = "wf_2_labels", masking_label_name = "nuclei")
>>> masked_label
>>> print(masked_label) # markdown-exec: hide
```
