"""Snippets for docs/tutorials/hcs_exploration.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/hcs_exploration.py
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


# --8<-- [start:open_plate]
from pathlib import Path

from ngio import open_ome_zarr_plate
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()

hcs_path = download_ome_zarr_dataset("CardiomyocyteTinyMip", download_dir=download_dir)
hcs_zarr = open_ome_zarr_plate(hcs_path)
print(hcs_zarr)
print(f"Rows: {hcs_zarr.rows}, Columns: {hcs_zarr.columns}")

# Get all the images in the plate
print(hcs_zarr.get_images())
# --8<-- [end:open_plate]

# --8<-- [start:concatenate_tables]
# Aggregate all table across all images
table = hcs_zarr.concatenate_image_tables(name="nuclei")
print_table(table.dataframe.head())
# --8<-- [end:concatenate_tables]

# --8<-- [start:save_table]
# Save the table in the HCS plate
hcs_zarr.add_table(name="nuclei", table=table, overwrite=True)

# Read the table back for sanity check
print_table(hcs_zarr.get_table("nuclei").dataframe.head())
# --8<-- [end:save_table]

# --8<-- [start:create_plate]
from ngio import ImageInWellPath, create_empty_plate

test_plate = create_empty_plate(
    store="./data/empty_plate.zarr",
    name="Test Plate",
    images=[
        ImageInWellPath(row="A", column="01", path="0"),
        ImageInWellPath(row="A", column="02", path="0"),
        ImageInWellPath(row="A", column="02", path="1", acquisition_id=1),
    ],
    overwrite=True,
)

print(test_plate)
print(f"Rows: {test_plate.rows}, Columns: {test_plate.columns}")
# --8<-- [end:create_plate]
