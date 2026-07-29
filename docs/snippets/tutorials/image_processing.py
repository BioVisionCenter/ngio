"""Snippets for docs/tutorials/image_processing.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/image_processing.py
"""

# --8<-- [start:plot_helpers]
from io import StringIO

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")


def print_figure(fig: Figure) -> None:
    """Print a figure as inline SVG, for markdown-exec `html` blocks.

    Recolours every bit of chrome to one sentinel, then swaps that sentinel for
    a theme variable in the emitted markup. The figure therefore follows the
    light/dark toggle rather than baking black text on white into the page.
    This only works because the SVG is inline: an `<img src>` would be a
    separate document and would not see the site's custom properties.

    The `.ngio-figure` wrapper is what the stylesheet keys on to strip the
    `OUT` terminal-output treatment that `.result` applies by default.
    """
    ink = "#5b6569"
    for ax in fig.axes:
        ax.tick_params(colors=ink, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(ink)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(ink)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(ink)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color(ink)
    for text in fig.texts:
        text.set_color(ink)

    buffer = StringIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)

    # Drop matplotlib's XML declaration and DOCTYPE — they are meaningless
    # inside an HTML body, and the figure is being embedded, not served.
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = svg.replace(ink, "var(--md-default-fg-color--light)")
    print(f'<div class="ngio-figure">{svg}</div>')


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

# Open the ome-zarr container
ome_zarr = open_ome_zarr_container(image_path)
# --8<-- [end:open_container]

# --8<-- [start:derive_image]
# First we will need the image object
image = ome_zarr.get_image()

# Second we need to derive a new ome-zarr image where we will store
# the processed image

blurred_omezarr_path = image_path.parent / "0_blurred"
blurred_omezarr = ome_zarr.derive_image(
    store=blurred_omezarr_path, name="Blurred Image", overwrite=True
)
blurred_image = blurred_omezarr.get_image()
# --8<-- [end:derive_image]

# --8<-- [start:apply_blur]
# We can use the axes order to specify how we query the image data.
# Here we will reorder the axes to be ["c", "z", "y", "x"].
# So that it will be compatible with the gaussian blur function
# which expects the channel axis to be the first one.
image_data = image.get_as_numpy(axes_order=["c", "z", "y", "x"])
# Apply gaussian blur to the image
sigma = 5.0
blurred_image_data = gaussian_blur(image_data, sigma=sigma)

# Set the processed image data back to the ome-zarr image
blurred_image.set_array(patch=blurred_image_data, axes_order=["c", "z", "y", "x"])

# The `set_array` method only saved the blurred image to the container at a specific
# resolution level. So all other resolution levels are still empty.
# To propagate the changes to all resolution levels,
# we can use the `consolidate` method.
blurred_image.consolidate()
# --8<-- [end:apply_blur]

# --8<-- [start:plot_blur]
fig, axs = plt.subplots(2, 1, figsize=(8, 4))
axs[0].set_title("Original image")
axs[0].imshow(image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]), cmap="gray")
axs[1].set_title("Blurred image")
axs[1].imshow(blurred_image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]), cmap="gray")
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_blur]

# --8<-- [start:dask_blur]
from dask import array as da


def dask_gaussian_blur(image: da.Array, sigma: float) -> da.Array:
    """Apply gaussian blur to a dask array."""
    # This will introduce some edge artifacts at chunk boundaries
    # In a real application, consider using map_overlap to mitigate this
    # With appropriate depth based on sigma
    return da.map_blocks(gaussian_blur, image, dtype=image.dtype, sigma=sigma)


image_dask = image.get_as_dask(axes_order=["c", "z", "y", "x"])
blurred_image_dask = dask_gaussian_blur(image_dask, sigma=sigma)
print(blurred_image_dask)
# --8<-- [end:dask_blur]

# --8<-- [start:iterators]
from ngio.experimental.iterators import ImageProcessingIterator

iterator = ImageProcessingIterator(
    input_image=image,
    output_image=blurred_image,
    axes_order=["c", "z", "y", "x"],
)

# After initializing the iterator, the iterator will have created
# will iterate over the entire image.
print(f"Iterator after initialization: {iterator}")

# Iterate over an arbitrary region of interest table
# We can use the product method that performs a cartesian product
# between the iterator and the table.
table = ome_zarr.get_roi_table("FOV_ROI_table")
iterator = iterator.product(table)
print(f"Iterator after product with table: {iterator}")

# We can explicitly set a broadcasting behavior
# For example we can iterate over all zyx planes, and broadcast all the other
# spatial dimensions
iterator = iterator.by_zyx()

# Finally (if needed) we can check if the regions are not-overlapping
iterator.require_no_regions_overlap()
# We can also check if the regions lay on non-overlapping chunks
iterator.require_no_chunks_overlap()

# Now we can map the gaussian blur function to the iterator
iterator.map_as_numpy(lambda x: gaussian_blur(x, sigma=sigma))

# No need to consolidate, the iterator takes care of that
# after all the regions have been processed
# --8<-- [end:iterators]
