"""Snippets for docs/tutorials/stitching.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/stitching.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, random_label_cmap, show_image
# --8<-- [end:plot_helpers]

# --8<-- [start:open_container]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteTinyMip", download_dir=download_dir)

# Open the OME-Zarr container
ome_zarr = open_ome_zarr_container(hcs_path / "B" / "03" / "0")
image = ome_zarr.get_image()
print(image)
# --8<-- [end:open_container]

# --8<-- [start:segmentation_fn]
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def segment(patch: np.ndarray) -> np.ndarray:
    # Smooth → Otsu threshold → distance transform → seeded watershed → cleanup
    smooth = ndi.gaussian_filter(patch, sigma=4)
    mask = smooth > threshold_otsu(smooth)
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=20, labels=mask)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    seg = watershed(-distance, markers, mask=mask).astype(np.uint32)
    # `max_size` drops objects at or below 500 px — skimage's renamed `min_size`.
    return remove_small_objects(seg, max_size=500)


# --8<-- [end:segmentation_fn]

# --8<-- [start:naive_tiling]
from ngio.iterators import SegmentationIterator
from ngio.transforms import UniqueLabelsTransform

naive = ome_zarr.derive_label("nuclei_tiled", overwrite=True)
tiling = SegmentationIterator(
    image, naive, channel_selection="DAPI", axes_order=["y", "x"]
).by_grid(size_x=512, size_y=512)

# Each tile numbers its objects from 1, so keeping them distinct in one array
# takes a per-tile offset (block i holds tile i's ids).
for i, roi in enumerate(tiling.rois):
    patch = image.get_roi_as_numpy(roi, c=0, axes_order=["y", "x"])
    naive.set_roi(
        roi,
        segment(patch),
        axes_order=["y", "x"],
        transforms=[UniqueLabelsTransform(10_000, i)],
    )
naive.consolidate(mode="auto")

print(f"tiles: {len(tiling.rois)}, ids: {len(np.unique(naive.get_as_numpy())) - 1}")
# --8<-- [end:naive_tiling]

# --8<-- [start:stitch]
stitched = ome_zarr.derive_label("nuclei_stitched", overwrite=True)
stitch_iterator = (
    SegmentationIterator(
        image,
        stitched,
        channel_selection="DAPI",
        axes_order=["y", "x"],
        consolidation_mode="auto",
    )
    .with_stitch()
    .by_grid(size_x=512, size_y=512)
    .with_halo(x=32, y=32)
)
stitch_iterator.map(segment)

print(f"objects: {len(np.unique(stitched.get_as_numpy())) - 1}")
# --8<-- [end:stitch]

# --8<-- [start:overview_figure]
cmap = random_label_cmap(n_labels=2000)
stitched_data = stitched.get_as_numpy(axes_order=["y", "x"])

image_data = image.get_as_numpy(c=0, axes_order=["y", "x"])
fig, (ax_img, ax_lab) = plt.subplots(2, 1, figsize=(8, 6.6))
show_image(ax_img, image_data, title="DAPI", pixel_size=image.pixel_size)
show_image(ax_lab, stitched_data, title="stitched segmentation", cmap=cmap)
for edge in range(512, image_data.shape[1], 512):
    ax_lab.axvline(edge - 0.5, color="white", ls="--", lw=0.5)
for edge in range(512, image_data.shape[0], 512):
    ax_lab.axhline(edge - 0.5, color="white", ls="--", lw=0.5)
print(
    figure_html(
        fig,
        alt="The DAPI image and its stitched segmentation, with the tile grid "
        "overlaid; no object shows a seam at a tile boundary.",
    )
)
# --8<-- [end:overview_figure]

# --8<-- [start:seam_figure]
# Found by eye; any seam tells the same story.
crop_y, crop_x = slice(1920, 2160), slice(1408, 1664)
seam_y, seam_x = 2048 - crop_y.start, 1536 - crop_x.start


def dense_ids(array: np.ndarray) -> np.ndarray:
    # Display-side renumbering only: the block-offset ids (1, 10_001, 20_001)
    # would otherwise collapse onto a handful of colormap entries.
    return np.unique(array, return_inverse=True)[1].reshape(array.shape)


naive_data = naive.get_as_numpy(axes_order=["y", "x"])
fig, (ax_raw, ax_naive, ax_merged) = plt.subplots(1, 3, figsize=(8.6, 3.1))
show_image(ax_raw, image_data[crop_y, crop_x], title="DAPI")
show_image(
    ax_naive,
    dense_ids(naive_data[crop_y, crop_x]),
    title="tiles segmented independently",
    cmap=cmap,
)
show_image(
    ax_merged,
    dense_ids(stitched_data[crop_y, crop_x]),
    title="with_stitch()",
    cmap=cmap,
)
for ax in (ax_naive, ax_merged):
    ax.axvline(seam_x - 0.5, color="white", ls="--", lw=1.0)
    ax.axhline(seam_y - 0.5, color="white", ls="--", lw=1.0)
print(
    figure_html(
        fig,
        alt="A zoom onto a corner where four tiles meet: segmented "
        "independently, the nucleus on the corner is split into one fragment "
        "per tile; with stitching it is one object.",
    )
)
# --8<-- [end:seam_figure]
