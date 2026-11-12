"""Snippets for docs/tutorials/image_segmentation.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/image_segmentation.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, random_label_cmap
# --8<-- [end:plot_helpers]

# --8<-- [start:segmentation_fn]
# Setup a simple segmentation function
import numpy as np
import skimage


def otsu_threshold_segmentation(image: np.ndarray) -> np.ndarray:
    """Simple segmentation using Otsu thresholding.

    Note there is no bookkeeping here: the function numbers its objects from
    1 like any segmenter, and keeping ids unique across regions is the
    iterator's job, not the function's.
    """
    threshold = skimage.filters.threshold_otsu(image)
    binary = image > threshold
    return skimage.measure.label(binary).astype(np.uint32)


# --8<-- [end:segmentation_fn]

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

# --8<-- [start:segment]
from ngio.iterators import SegmentationIterator

# Take the image to read from, and the FOV table naming the regions to walk
image = ome_zarr.get_image()
roi_table = ome_zarr.get_roi_table("FOV_ROI_table")

# Derive an empty label image to write the segmentation into
label = ome_zarr.derive_label("new_label", overwrite=True)

# Setup the segmentation iterator
seg_iterator = SegmentationIterator(
    input_image=image,
    output_label=label,
    channel_selection="DAPI",
    axes_order=["z", "y", "x"],
    consolidation_mode="auto",
)
seg_iterator = seg_iterator.with_stitch().product(roi_table)

# Split any remaining time axis, so each step yields one whole ZYX volume
seg_iterator = seg_iterator.by_zyx()

# Each FOV reads a halo past its edge; `with_stitch()` uses that overlap to give
# every FOV its own id block and to merge objects split by a FOV boundary,
# then renumbers everything to a dense 1..N.
seg_iterator = seg_iterator.with_halo(x=16, y=16)

seg_iterator.map(otsu_threshold_segmentation)

# No need to consolidate, the iterator does it automatically after the map
# --8<-- [end:segment]

# --8<-- [start:plot_segmentation]
rand_cmap = random_label_cmap()
original = image.get_as_numpy(c=0, z=1, axes_order=["y", "x"])

# The data does not fill its uint16 range, so window it on percentiles.
vmin, vmax = np.percentile(original, (1, 99.8))

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title("Original image")
axs[0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
axs[1].set_title("Final segmentation")
axs[1].imshow(
    label.get_as_numpy(z=1, axes_order=["y", "x"]),
    cmap=rand_cmap,
    interpolation="nearest",
)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print(figure_html(fig))
# --8<-- [end:plot_segmentation]

# --8<-- [start:create_mask]
# Create a basic mask for illustration purposes
mask = ome_zarr.derive_label("mask", overwrite=True)
mask_data = mask.get_as_numpy(axes_order=["z", "y", "x"])
mask_data[:, 200:-200, 500:2000] = 1
mask_data[:, 200:-200, 3000:-500] = 2
mask_data[:, 600:-600, 1200:-1000] = 0
mask_data[:, 700:-700, 1600:-1500] = 3
mask.set_array(mask_data, axes_order=["z", "y", "x"])
mask.consolidate(mode="auto")
# --8<-- [end:create_mask]

# --8<-- [start:plot_mask]
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title("Original image")
axs[0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
axs[1].set_title("Mask")
axs[1].imshow(
    mask.get_as_numpy(z=1, axes_order=["y", "x"]),
    cmap=rand_cmap,
    interpolation="nearest",
)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print(figure_html(fig))
# --8<-- [end:plot_mask]

# --8<-- [start:masked_segment]
from ngio.iterators import MaskedSegmentationIterator

# Take a masked image, which carries its masking ROI table with it
image = ome_zarr.get_masked_image(masking_label_name="mask")

# Derive an empty label image to write the segmentation into
label = ome_zarr.derive_label("masked_new_label", overwrite=True)

# Setup the masked segmentation iterator
from ngio.transforms import UniqueLabelsTransform

# Each mask's ids land in a block of their own — the ROI's label picks the
# block, so the transform needs no per-region state and the map can even run
# under a parallel mapper.
seg_iterator = MaskedSegmentationIterator(
    input_image=image,
    output_label=label,
    channel_selection="DAPI",
    axes_order=["z", "y", "x"],
    consolidation_mode="auto",
    output_transforms=[UniqueLabelsTransform(10_000)],
)

# Split any remaining time axis, so each step yields one whole ZYX volume
seg_iterator = seg_iterator.by_zyx()

seg_iterator.map(otsu_threshold_segmentation)

# No need to consolidate, the iterator does it automatically after the map
# --8<-- [end:masked_segment]

# --8<-- [start:plot_masked_segmentation]
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title("Original image")
axs[0].imshow(
    image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]),
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
axs[1].set_title("Final segmentation")
axs[1].imshow(
    label.get_as_numpy(z=1, axes_order=["y", "x"]),
    cmap=rand_cmap,
    interpolation="nearest",
)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print(figure_html(fig))
# --8<-- [end:plot_masked_segmentation]
