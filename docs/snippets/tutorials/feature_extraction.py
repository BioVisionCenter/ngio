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

from _render import table_html
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

# --8<-- [start:read_table_back]
print(table_html(ome_zarr.get_table("nuclei_regionprops").dataframe.head()))
# --8<-- [end:read_table_back]
