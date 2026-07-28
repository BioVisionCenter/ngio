"""Snippets for docs/tutorials/hcs_exploration.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/hcs_exploration.py
"""

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
print(table.dataframe.head().to_markdown())
# --8<-- [end:concatenate_tables]

# --8<-- [start:save_table]
# Save the table in the HCS plate
hcs_zarr.add_table(name="nuclei", table=table, overwrite=True)

# Read the table back for sanity check
print(hcs_zarr.get_table("nuclei").dataframe.head().to_markdown())
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
