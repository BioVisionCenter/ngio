"""Snippets for docs/tutorials/hcs_exploration.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/hcs_exploration.py
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
print(hcs_zarr.get_images(max_workers="auto"))
# --8<-- [end:open_plate]

# --8<-- [start:concatenate_tables]
# Aggregate all table across all images
table = hcs_zarr.concatenate_image_tables(name="nuclei", max_workers="auto")
print(table_html(table.dataframe.head()))
# --8<-- [end:concatenate_tables]

# --8<-- [start:save_table]
# Save the table in the HCS plate
hcs_zarr.add_table(name="nuclei", table=table, overwrite=True)

# Read the table back for sanity check
print(table_html(hcs_zarr.get_table("nuclei").dataframe.head()))
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
