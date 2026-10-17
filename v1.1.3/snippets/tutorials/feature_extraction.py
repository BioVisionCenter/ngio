"""Snippets for docs/tutorials/feature_extraction.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/feature_extraction.py
"""

# --8<-- [start:table_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, table_html
# --8<-- [end:table_helpers]


# --8<-- [start:extract_features]
import numpy as np
import pandas as pd
from skimage import measure

from ngio import Roi


def extract_features(image: np.ndarray, label: np.ndarray, roi: Roi) -> pd.DataFrame:
    """Basic feature extraction using skimage.measure.regionprops_table."""
    label = label.squeeze(-1)  # Remove the channel axis if present
    roi_feat_table = measure.regionprops_table(
        label_image=label,
        intensity_image=image,
        properties=[
            "label",
            "area",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ],
    )
    return pd.DataFrame(roi_feat_table)


# --8<-- [end:extract_features]

# --8<-- [start:open_container]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteTinyMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr = open_ome_zarr_container(image_path)
# --8<-- [end:open_container]

# --8<-- [start:setup_transform]
from ngio.transforms import ZoomTransform

# Take the image to measure
image = ome_zarr.get_image()

# Get the nuclei label
nuclei = ome_zarr.get_label("nuclei")

# Here the image is stored at a higher resolution than the nuclei label
print(f"Image dimensions: {image.dimensions}, pixel size: {image.pixel_size}")
print(f"Nuclei dimensions: {nuclei.dimensions}, pixel size: {nuclei.pixel_size}")

# So resample the label up to the image resolution with a transform
zoom_transform = ZoomTransform(
    input_image=nuclei,
    target_image=image,
    order="nearest",  # Nearest-neighbour interpolation, so label ids stay intact
)
# --8<-- [end:setup_transform]

# --8<-- [start:extract]
from ngio.iterators import FeatureExtractorIterator

iterator = FeatureExtractorIterator(
    input_image=image,
    input_label=nuclei,
    label_transforms=[zoom_transform],
    axes_order=["y", "x", "c"],
)

# Measure every region and join the per-region results into ONE FeatureTable.
# Pass `mapper=ThreadedMapper("auto")` to fan the measurements out in parallel;
# the join always happens once, at the end. Nothing is written yet.
feat_table = iterator.measure(extract_features)
assert feat_table is not None  # a serial run always returns the table

# Storing the table is a separate, explicit step.
ome_zarr.add_table("nuclei_regionprops", feat_table, overwrite=True)
# --8<-- [end:extract]

# --8<-- [start:manual_extract]
from ngio.tables import FeatureTable

feat_frames = []
for image_data, label_data, roi in iterator.iter_as_numpy():
    print(f"Processing ROI: {roi}")
    feat_frames.append(extract_features(image_data, label_data, roi))

# Concatenate the per-region frames into one table
manual_table = FeatureTable(table_data=pd.concat(feat_frames), reference_label="nuclei")
ome_zarr.add_table("nuclei_regionprops_manual", manual_table, overwrite=True)
# --8<-- [end:manual_extract]

# --8<-- [start:halo_dedup]
from ngio.tables import Table


def keep_most_complete(results: list[pd.DataFrame]) -> Table:
    """One row per object: the measurement from the tile that saw most of it."""
    joined = pd.concat([frame for frame in results if len(frame)])
    joined = (
        joined.sort_values("area", ascending=False)
        .drop_duplicates("label")
        .drop(columns=["roi_index", "roi_name"])
        .set_index("label")
        .sort_index()
    )
    return FeatureTable(table_data=joined, reference_label="nuclei")


# Four tiles, each reading 32 px of context past its edges: a border nucleus
# is measured whole by every tile that sees it, so its label shows up more
# than once — each row stamped with the `roi_index`/`roi_name` it came from.
tiled = iterator.by_blocks(num_y=2, num_x=2).with_halo(y=32, x=32)
tiled_table = tiled.with_join(keep_most_complete).measure(extract_features)
assert tiled_table is not None
ome_zarr.add_table("nuclei_regionprops_tiled", tiled_table, overwrite=True)
# --8<-- [end:halo_dedup]

# --8<-- [start:read_table_back]
print(table_html(ome_zarr.get_table("nuclei_regionprops").dataframe.head()))
# --8<-- [end:read_table_back]

# --8<-- [start:plot_features]
df = ome_zarr.get_table("nuclei_regionprops").dataframe
area_um2 = df["area"] * image.pixel_size.x**2

fig, ax = plt.subplots(figsize=(6.4, 4.2))
# rasterized=True bakes the ~1500 dots into a small embedded image (set_dpi keeps
# them crisp); the axes and labels stay vector, so the figure embeds light.
ax.scatter(
    area_um2,
    df["mean_intensity-0"],
    s=10,
    alpha=0.45,
    color="#22a699",
    linewidths=0,
    rasterized=True,
)
ax.set_xlabel("nucleus area (µm²)")
ax.set_ylabel("mean DAPI intensity")
ax.set_title("one dot per nucleus")
ax.spines[["top", "right"]].set_visible(False)
fig.set_dpi(200)
print(
    figure_html(
        fig,
        alt="Scatter of nucleus area against mean DAPI intensity: one dense "
        "cloud around 130 square micrometers, and a tail of small, bright "
        "nuclei in the upper left.",
    )
)
# --8<-- [end:plot_features]
