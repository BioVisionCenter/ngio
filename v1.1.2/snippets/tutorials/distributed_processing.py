"""Snippets for docs/tutorials/distributed_processing.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/distributed_processing.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, show_image, table_html
# --8<-- [end:plot_helpers]

# --8<-- [start:open_container]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteTinyMip", download_dir=download_dir)

# Open the OME-Zarr container. The store is on disk — a distributed run cannot
# use an in-memory store, since each process would write its own private copy.
ome_zarr = open_ome_zarr_container(hcs_path / "B" / "03" / "0")
image = ome_zarr.get_image()
# --8<-- [end:open_container]

# --8<-- [start:segmentation_fn]
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


# Same watershed pipeline as docs/snippets/tutorials/stitching.py; each snippet
# script stands alone, so the function is repeated rather than imported.
def segment(patch: np.ndarray) -> np.ndarray:
    # Smooth → Otsu threshold → distance transform → seeded watershed → cleanup
    smooth = ndi.gaussian_filter(patch, sigma=4)
    mask = smooth > threshold_otsu(smooth)
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=20, labels=mask)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    seg = watershed(-distance, markers, mask=mask).astype(np.uint32)
    return remove_small_objects(seg, max_size=500)


# --8<-- [end:segmentation_fn]

# --8<-- [start:fat_partitions]
from ngio.iterators import SegmentationIterator

# Deriving with no arguments inherits the image's chunking: two fat chunks.
label = ome_zarr.derive_label("nuclei_distributed", overwrite=True)
print(f"write granularity: {label.write_granularity}")

iterator = SegmentationIterator(
    image, label, channel_selection="DAPI", axes_order=["y", "x"]
).by_grid(size_x=512, size_y=512)

layout = [iterator.for_job(i, n_jobs=4).partition_indices for i in range(4)]
print(f"tiles per job: {[len(part) for part in layout]}")
# --8<-- [end:fat_partitions]

# --8<-- [start:aligned_partitions]
# Chunk the output like the tiling (chunks are given in the reference image's
# axes, here czyx; the channel axis is squeezed away on a label).
label = ome_zarr.derive_label(
    "nuclei_distributed", chunks=(1, 1, 512, 512), overwrite=True
)
print(f"write granularity: {label.write_granularity}")

# `by_write_units` tiles by exactly that granularity, so every tile is its own
# independent write unit.
iterator = SegmentationIterator(
    image, label, channel_selection="DAPI", axes_order=["y", "x"]
).by_write_units()

layout = [iterator.for_job(i, n_jobs=4).partition_indices for i in range(4)]
print(f"tiles per job: {[len(part) for part in layout]}")
# --8<-- [end:aligned_partitions]

# --8<-- [start:partition_figure]
from matplotlib.colors import ListedColormap

image_data = image.get_as_numpy(c=0, axes_order=["y", "x"])
job_map = np.full(image_data.shape, np.nan)
for job, indices in enumerate(layout):
    for index in indices:
        box = iterator.rois[index].to_pixel(pixel_size=image.pixel_size)
        y, x = box["y"], box["x"]
        job_map[
            int(y.start) : int(y.start + y.length),
            int(x.start) : int(x.start + x.length),
        ] = job

fig, ax = plt.subplots(figsize=(8, 3.4))
show_image(ax, image_data, title="which job owns which write unit (n_jobs=4)")
job_colors = ListedColormap(["#2e6fd6", "#22a699", "#f4a63a", "#c2185b"])
ax.imshow(job_map, cmap=job_colors, alpha=0.35, interpolation="nearest")
for edge in range(512, image_data.shape[1], 512):
    ax.axvline(edge - 0.5, color="white", lw=0.4)
for edge in range(512, image_data.shape[0], 512):
    ax.axhline(edge - 0.5, color="white", lw=0.4)
print(
    figure_html(
        fig,
        alt="The image tiled into 50 write units, each tinted by the job that "
        "owns it; the four jobs interleave freely because no two of them "
        "ever share a write unit.",
    )
)
# --8<-- [end:partition_figure]


# --8<-- [start:build_iterator]
def build_iterator() -> SegmentationIterator:
    # Rebuilt identically in every phase; construction is metadata-only.
    return (
        SegmentationIterator(
            image,
            ome_zarr.get_label("nuclei_distributed"),
            channel_selection="DAPI",
            axes_order=["y", "x"],
            consolidation_mode="auto",
        )
        .with_stitch()
        .by_write_units()
        .with_halo(x=32, y=32)
    )


# --8<-- [end:build_iterator]

# --8<-- [start:init_task]
args_list = build_iterator().prepare_jobs(n_jobs=4)
print(args_list)
# --8<-- [end:init_task]

# --8<-- [start:parallel_tasks]
# On a real cluster each iteration is its own scheduler task.
for args in args_list:
    build_iterator().for_job(**args).segment(segment)
# --8<-- [end:parallel_tasks]

# --8<-- [start:consolidate_task]
build_iterator().finalize()

final = ome_zarr.get_label("nuclei_distributed")
print(f"objects: {len(np.unique(final.get_as_numpy())) - 1}")
# --8<-- [end:consolidate_task]

# --8<-- [start:measure_fn]
import pandas as pd
from skimage.measure import regionprops_table

from ngio import Roi


def measure(image_data: np.ndarray, label_data: np.ndarray, roi: Roi) -> pd.DataFrame:
    props = regionprops_table(
        label_image=label_data.squeeze(-1),  # remove the channel axis
        intensity_image=image_data,
        properties=["label", "area", "mean_intensity"],
    )
    return pd.DataFrame(props)


# --8<-- [end:measure_fn]

# --8<-- [start:measure_distributed]
from ngio.iterators import FeatureExtractorIterator


def build_measure_iterator() -> FeatureExtractorIterator:
    return FeatureExtractorIterator(
        image,
        ome_zarr.get_label("nuclei_distributed"),
        axes_order=["y", "x", "c"],
    ).by_blocks(num_x=2, num_y=2)


# init task
measure_args = build_measure_iterator().prepare_jobs(n_jobs=2)

# parallel tasks: on a slice, `measure` banks a partial and returns None
for args in measure_args:
    build_measure_iterator().for_job(**args).measure(measure)

# consolidate task: the one global join returns the table; storing it is yours
table = build_measure_iterator().finalize()
ome_zarr.add_table("nuclei_features_distributed", table, overwrite=True)
print(f"rows: {len(table.dataframe)}")
# --8<-- [end:measure_distributed]

# --8<-- [start:read_table_back]
print(table_html(ome_zarr.get_table("nuclei_features_distributed").dataframe.head()))
# --8<-- [end:read_table_back]
