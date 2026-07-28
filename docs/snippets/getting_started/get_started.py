"""Snippets for docs/getting_started/1_ome_zarr_containers.md, 2_images.md, 3_tables.md.

These three pages share the `get_started` markdown-exec session, so they share this
one script and it stays runnable on its own:

    python docs/snippets/getting_started/get_started.py

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into a page by `pymdownx.snippets` and executed by `markdown-exec`. Sections follow
page order, and several of them rebind names used by later sections, so the order
here is load-bearing.
"""

# --8<-- [start:plot_helpers]
from io import StringIO

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

matplotlib.use("Agg")


def random_label_cmap(n_labels: int = 1000, seed: int = 0) -> ListedColormap:
    """Build a reproducible random colormap for label images."""
    rng = np.random.default_rng(seed)
    colors = rng.random((n_labels, 3))
    colors[0] = 0.0
    return ListedColormap(colors)


def print_figure(fig: Figure) -> None:
    """Print a figure as inline SVG, for markdown-exec `html` blocks."""
    buffer = StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    print(buffer.getvalue())


# --8<-- [end:plot_helpers]

# ---------------------------------------------------------------------------
# 1. OME-Zarr Container
# ---------------------------------------------------------------------------

# --8<-- [start:setup]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr_container = open_ome_zarr_container(image_path)
# --8<-- [end:setup]

# --8<-- [start:print_container]
print(ome_zarr_container)
# --8<-- [end:print_container]

# --8<-- [start:levels]
print(ome_zarr_container.levels)  # Show the number of resolution levels
# --8<-- [end:levels]

# --8<-- [start:level_paths]
print(ome_zarr_container.level_paths)  # Show the paths to all available images
# --8<-- [end:level_paths]

# --8<-- [start:is_3d]
print(ome_zarr_container.is_3d)  # Get if the image is 3D
# --8<-- [end:is_3d]

# --8<-- [start:is_time_series]
print(ome_zarr_container.is_time_series)  # Get if the image is a time series
# --8<-- [end:is_time_series]

# --8<-- [start:metadata]
metadata = ome_zarr_container.meta
print(metadata)
# --8<-- [end:metadata]

# --8<-- [start:channel_labels]
print(metadata.channels_meta.channel_labels)
# --8<-- [end:channel_labels]

# ---------------------------------------------------------------------------
# 2. Images and Labels
# ---------------------------------------------------------------------------

# --8<-- [start:get_image_default]
print(ome_zarr_container.get_image())  # Get the highest resolution image
# --8<-- [end:get_image_default]

# --8<-- [start:get_image_by_path]
print(ome_zarr_container.get_image(path="1"))  # Get a specific pyramid level
# --8<-- [end:get_image_by_path]

# --8<-- [start:get_image_by_pixel_size]
from ngio import PixelSize

pixel_size = PixelSize(x=0.65, y=0.65, z=1.0)
image = ome_zarr_container.get_image(pixel_size=pixel_size)
print(image)
# --8<-- [end:get_image_by_pixel_size]

# --8<-- [start:get_image_nearest]
from ngio import PixelSize

pixel_size = PixelSize(x=0.60, y=0.60, z=1.0)
image = ome_zarr_container.get_image(pixel_size=pixel_size, strict=False)
print(image)
# --8<-- [end:get_image_nearest]

# --8<-- [start:image_dimensions]
print(image.dimensions)
# --8<-- [end:image_dimensions]

# --8<-- [start:image_pixel_size]
print(image.pixel_size)
# --8<-- [end:image_pixel_size]

# --8<-- [start:image_array_info]
print(image.shape, image.dtype, image.chunks)
# --8<-- [end:image_array_info]

# --8<-- [start:image_as_numpy]
data = image.get_as_numpy()  # Get the image as a numpy array
print(data.shape, data.dtype)
# --8<-- [end:image_as_numpy]

# --8<-- [start:image_as_dask]
dask_array = image.get_as_dask()  # Get the image as a dask array
print(dask_array)
# --8<-- [end:image_as_dask]

# --8<-- [start:image_get_array_legacy]
# Get the image as a numpy or dask or delayed object
data = image.get_array(mode="numpy")
print(data.shape, data.dtype)
# --8<-- [end:image_get_array_legacy]

# --8<-- [start:image_slice]
# Get a specific channel and axes order
image_slice = image.get_as_numpy(
    channel_selection="DAPI",
    x=slice(0, 128),
    axes_order=["t", "z", "y", "x", "c"],
)
print(image_slice.shape)
# --8<-- [end:image_slice]

# --8<-- [start:set_array_example]
import numpy as np


def process(patch: np.ndarray) -> np.ndarray:
    """Placeholder for your own processing step.

    Replace the body with the operation you want to apply to the patch.
    """
    return patch


# Get the image data as a numpy array
data = image.get_as_numpy(
    channel_selection="DAPI",
    x=slice(0, 128),
    y=slice(0, 128),
    axes_order=["z", "y", "x", "c"],
)

# Modify the image data
data = process(data)

# Set the modified image data
image.set_array(
    data,
    channel_selection="DAPI",
    x=slice(0, 128),
    y=slice(0, 128),
    axes_order=["z", "y", "x", "c"],
)

# Consolidate the changes to all resolution levels, see below for more details
image.consolidate()
# --8<-- [end:set_array_example]

# --8<-- [start:roi_slicing]
from ngio import Roi

# Define a ROI in world coordinates
roi = Roi.from_values(slices={"x": (34.1, 321.6), "y": (10, 330)}, name=None)
# Get the image data in the ROI as a numpy array
print(image.get_roi_as_numpy(roi).shape)
# --8<-- [end:roi_slicing]

# --8<-- [start:list_labels]
print(ome_zarr_container.list_labels())  # Available labels
# --8<-- [end:list_labels]

# --8<-- [start:get_label_default]
# Get the highest resolution label
print(ome_zarr_container.get_label("nuclei"))
# --8<-- [end:get_label_default]

# --8<-- [start:get_label_by_path]
# Get a specific pyramid level
print(ome_zarr_container.get_label("nuclei", path="1"))
# --8<-- [end:get_label_by_path]

# --8<-- [start:get_label_by_pixel_size]
from ngio import PixelSize

pixel_size = PixelSize(x=0.65, y=0.65, z=1.0)
label_nuclei = ome_zarr_container.get_label("nuclei", pixel_size=pixel_size)
print(label_nuclei)
# --8<-- [end:get_label_by_pixel_size]

# --8<-- [start:get_label_nearest]
from ngio import PixelSize

pixel_size = PixelSize(x=0.60, y=0.60, z=1.0)
label_nuclei = ome_zarr_container.get_label(
    "nuclei", pixel_size=pixel_size, strict=False
)
print(label_nuclei)
# --8<-- [end:get_label_nearest]

# --8<-- [start:derive_label]
# Derive a new label
new_label = ome_zarr_container.derive_label("new_label", overwrite=True)
print(new_label)
# --8<-- [end:derive_label]

# ---------------------------------------------------------------------------
# 3. Tables
# ---------------------------------------------------------------------------

# --8<-- [start:list_tables]
# List all available tables
print(ome_zarr_container.list_tables())
# --8<-- [end:list_tables]

# --8<-- [start:roi_table_get]
roi_table = ome_zarr_container.get_table("FOV_ROI_table")  # Get a ROI table
print(roi_table.get("FOV_1"))
# --8<-- [end:roi_table_get]

# --8<-- [start:plot_fov_roi_on_image]
from matplotlib.patches import Rectangle

image_3 = ome_zarr_container.get_image(path="3")
image_data = np.squeeze(image_3.get_as_numpy(c=0))

roi = roi_table.get("FOV_1").to_pixel(pixel_size=image_3.pixel_size)
x_slice = roi.get("x")
y_slice = roi.get("y")

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("FOV_1 ROI")
ax.imshow(image_data, cmap="gray")
ax.add_patch(
    Rectangle(
        (x_slice.start, y_slice.start),
        x_slice.length,
        y_slice.length,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
)
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_fov_roi_on_image]

# --8<-- [start:roi_table_slice_image]
roi = roi_table.get("FOV_1")
roi_data = image.get_roi_as_numpy(roi)
print(roi_data.shape)
# --8<-- [end:roi_table_slice_image]

# --8<-- [start:plot_fov_roi_crop]
roi = roi_table.get("FOV_1")
image_3 = ome_zarr_container.get_image(path="3")
image_data = np.squeeze(image_3.get_roi_as_numpy(roi, c=0))

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("FOV_1 ROI")
ax.imshow(image_data, cmap="gray")
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_fov_roi_crop]

# --8<-- [start:masking_table_get]
# Get a mask table
masking_table = ome_zarr_container.get_table("nuclei_ROI_table")
print(masking_table.get_label(100))
# --8<-- [end:masking_table_get]

# --8<-- [start:masking_table_slice_image]
roi = masking_table.get_label(100)
roi_data = image.get_roi_as_numpy(roi)
print(roi_data.shape)
# --8<-- [end:masking_table_slice_image]

# --8<-- [start:plot_masking_roi_crop]
cmap = random_label_cmap()

roi = masking_table.get_label(100)
image_2 = ome_zarr_container.get_image(path="2")
image_data = np.squeeze(image_2.get_roi_as_numpy(roi, c=0))

label_2 = ome_zarr_container.get_label("nuclei", pixel_size=image_2.pixel_size)
label_data = np.squeeze(label_2.get_roi_as_numpy(roi))

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Label 100 ROI")
ax.imshow(image_data, cmap="gray")
ax.imshow(label_data, cmap=cmap, alpha=0.6)
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_masking_roi_crop]

# --8<-- [start:feature_table]
# Get a feature table
feature_table = ome_zarr_container.get_table("regionprops_DAPI")
# only show the first 5 rows
print(feature_table.dataframe.head(5).to_markdown())
# --8<-- [end:feature_table]

# --8<-- [start:create_roi_table]
from ngio import Roi
from ngio.tables import RoiTable

roi = Roi.from_values(slices={"x": (0, 128), "y": (0, 128)}, name="FOV_1")
roi_table = RoiTable(rois=[roi])
print(roi_table)
# --8<-- [end:create_roi_table]

# --8<-- [start:build_image_roi_table]
roi_table = ome_zarr_container.build_image_roi_table("whole_image")
print(roi_table)
# --8<-- [end:build_image_roi_table]

# --8<-- [start:add_roi_table]
ome_zarr_container.add_table("new_roi_table", roi_table, overwrite=True)
roi_table = ome_zarr_container.get_table("new_roi_table")
print(roi_table)
# --8<-- [end:add_roi_table]

# --8<-- [start:build_masking_roi_table]
masking_table = ome_zarr_container.build_masking_roi_table("nuclei")
print(masking_table)
# --8<-- [end:build_masking_roi_table]

# --8<-- [start:create_feature_table]
import pandas as pd

from ngio.tables import FeatureTable

example_data = pd.DataFrame({"label": [1, 2, 3], "area": [100, 200, 300]})
feature_table = FeatureTable(table_data=example_data)
print(feature_table)
# --8<-- [end:create_feature_table]

# --8<-- [start:create_generic_table]
import pandas as pd

from ngio.tables import GenericTable

example_data = pd.DataFrame({"area": [100, 200, 300], "perimeter": [50, 60, 70]})
generic_table = GenericTable(table_data=example_data)
print(generic_table)
# --8<-- [end:create_generic_table]

# --8<-- [start:generic_table_from_anndata]
import anndata as ad
import numpy as np
import pandas as pd

from ngio.tables import GenericTable

adata = ad.AnnData(
    X=np.random.rand(10, 5),
    obs=pd.DataFrame({"cell_type": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]}),
)
generic_table = GenericTable(table_data=adata)
print(generic_table)
# --8<-- [end:generic_table_from_anndata]
