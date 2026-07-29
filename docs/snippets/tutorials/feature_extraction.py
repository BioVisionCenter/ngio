"""Snippets for docs/tutorials/feature_extraction.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/feature_extraction.py
"""

# --8<-- [start:table_helpers]
import pandas as pd


def print_table(df: pd.DataFrame) -> None:
    """Print a DataFrame as HTML that the docs theme will style.

    Markdown is not an option here: Zensical does not run block-level Markdown over
    markdown-exec output, so a pipe table would stay literal text. The theme styles
    only `table:not([class])` — and its JS only wraps such tables in a horizontal
    scroll container — while pandas tags its output `class="dataframe"`, so the class
    and the presentational border are stripped.
    """
    # A named index (here the label id) is real data, so promote it to a column: pandas
    # otherwise renders it as a second, near-empty header row.
    if df.index.name is not None:
        df = df.reset_index()
    html = df.to_html(index=False, border=0, float_format="{:.2f}".format)
    print(html.replace(' class="dataframe"', ""))


# --8<-- [end:table_helpers]


# --8<-- [start:extract_features]
import numpy as np
import pandas as pd
from skimage import measure


def extract_features(image: np.ndarray, label: np.ndarray) -> pd.DataFrame:
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

# Open the ome-zarr container
ome_zarr = open_ome_zarr_container(image_path)
# --8<-- [end:open_container]

# --8<-- [start:setup_transform]
from ngio.transforms import ZoomTransform

# First we will need the image object and the FOVs table
image = ome_zarr.get_image()

# Get the nuclei label
nuclei = ome_zarr.get_label("nuclei")

# In this example we the image is available at an higher resolution than the nuclei
print(f"Image dimensions: {image.dimensions}, pixel size: {image.pixel_size}")
print(f"Nuclei dimensions: {nuclei.dimensions}, pixel size: {nuclei.pixel_size}")

# We need to setup a transform to resample the nuclei to the image resolution
zoom_transform = ZoomTransform(
    input_image=nuclei,
    target_image=image,
    order="nearest",  # Nearest neighbor interpolation for labels
)
# --8<-- [end:setup_transform]

# --8<-- [start:extract]
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable

iterator = FeatureExtractorIterator(
    input_image=image,
    input_label=nuclei,
    label_transforms=[zoom_transform],
    axes_order=["y", "x", "c"],
)

feat_table = []
for image_data, label_data, roi in iterator.iter_as_numpy():
    print(f"Processing ROI: {roi}")
    roi_feat_table = extract_features(image=image_data, label=label_data)
    feat_table.append(roi_feat_table)

# Concatenate all the dataframes into a single one
feat_table = pd.concat(feat_table)
feat_table = FeatureTable(table_data=feat_table, reference_label="nuclei")
ome_zarr.add_table("nuclei_regionprops", feat_table, overwrite=True)
# --8<-- [end:extract]

# --8<-- [start:read_table_back]
print_table(ome_zarr.get_table("nuclei_regionprops").dataframe.head())
# --8<-- [end:read_table_back]
