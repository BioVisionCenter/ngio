"""Snippets for docs/tutorials/image_processing.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/image_processing.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html
# --8<-- [end:plot_helpers]

# --8<-- [start:gaussian_blur]
import numpy as np
import skimage


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply gaussian blur to an image."""
    original_type = image.dtype
    image = skimage.filters.gaussian(
        image, sigma=sigma, channel_axis=0, preserve_range=True
    )
    # Convert the image back to the original type
    image = image.astype(original_type)
    return image


# --8<-- [end:gaussian_blur]

# --8<-- [start:open_container]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()

hcs_path = download_ome_zarr_dataset("CardiomyocyteTiny", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr = open_ome_zarr_container(image_path)
# --8<-- [end:open_container]

# --8<-- [start:derive_image]
# Take the image to read from
image = ome_zarr.get_image()

# Derive a new OME-Zarr container to store the processed image in

blurred_omezarr_path = image_path.parent / "0_blurred"
blurred_omezarr = ome_zarr.derive_image(
    store=blurred_omezarr_path, name="Blurred Image", overwrite=True
)
blurred_image = blurred_omezarr.get_image()
# --8<-- [end:derive_image]

# --8<-- [start:apply_blur]
# `axes_order` sets the order the array comes back in. Here it is
# ["c", "z", "y", "x"], to match the blur function, which expects the
# channel axis first.
image_data = image.get_as_numpy(axes_order=["c", "z", "y", "x"])
# Apply gaussian blur to the image
sigma = 5.0
blurred_image_data = gaussian_blur(image_data, sigma=sigma)

# Write the processed data back to the OME-Zarr image
blurred_image.set_array(patch=blurred_image_data, axes_order=["c", "z", "y", "x"])

# `set_array` wrote to one resolution level only, so the rest of the pyramid is
# still empty. `consolidate` rebuilds the other levels from it.
blurred_image.consolidate()
# --8<-- [end:apply_blur]

# --8<-- [start:plot_blur]
original = image.get_as_numpy(c=0, z=1, axes_order=["y", "x"])
blurred = blurred_image.get_as_numpy(c=0, z=1, axes_order=["y", "x"])

# The data does not fill its uint16 range, so window it on percentiles. One window for
# both panels: stretching them separately would misrepresent the difference.
vmin, vmax = np.percentile(original, (1, 99.8))

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title("Original image")
axs[0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
axs[1].set_title("Blurred image")
axs[1].imshow(blurred, cmap="gray", vmin=vmin, vmax=vmax)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print(figure_html(fig))
# --8<-- [end:plot_blur]

# --8<-- [start:dask_blur]
from dask import array as da


def dask_gaussian_blur(image: da.Array, sigma: float) -> da.Array:
    """Apply gaussian blur to a dask array."""
    # This introduces edge artefacts at chunk boundaries. In a real application,
    # use map_overlap with a depth chosen from sigma to avoid them.
    return da.map_blocks(gaussian_blur, image, dtype=image.dtype, sigma=sigma)


image_dask = image.get_as_dask(axes_order=["c", "z", "y", "x"])
blurred_image_dask = dask_gaussian_blur(image_dask, sigma=sigma)
print(blurred_image_dask)
# --8<-- [end:dask_blur]

# --8<-- [start:iterators]
from ngio.iterators import ImageProcessingIterator

iterator = ImageProcessingIterator(
    input_image=image,
    output_image=blurred_image,
    axes_order=["c", "z", "y", "x"],
)

# A freshly built iterator covers the entire image in one region.
print(f"Iterator over the whole image: {iterator}")

# Narrow it to an arbitrary ROI table. `product` takes the cartesian product
# of the iterator's regions and the table's.
table = ome_zarr.get_roi_table("FOV_ROI_table")
iterator = iterator.product(table)
print(f"Iterator after product with table: {iterator}")

# Set the broadcasting explicitly. `by_zyx` splits the time axis, so each step
# yields one whole ZYX volume rather than the full time series at once.
iterator = iterator.by_zyx()

# Optionally assert the regions do not overlap each other...
iterator.require_no_regions_overlap()
# ...nor share chunks, which is what makes parallel writes safe.
iterator.require_no_write_units_overlap()

# A blur at a region edge has no neighbours to average with, which is exactly
# the seam artifact the dask version above warns about. `with_halo` fixes it:
# each region reads a margin of context, and the border is cropped before the
# write, so the write footprints (and the parallel safety) are unchanged.
iterator = iterator.with_halo(x=16, y=16)

# Map the blur across every region, fanning out on a thread pool.
from ngio import ThreadedMapper

iterator.map(lambda x: gaussian_blur(x, sigma=sigma), mapper=ThreadedMapper("auto"))

# No need to consolidate: the iterator does it once every region is processed
# --8<-- [end:iterators]
