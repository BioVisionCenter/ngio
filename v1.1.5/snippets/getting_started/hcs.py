"""Snippets for docs/getting_started/5_hcs.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/hcs.py
"""

# --8<-- [start:setup]
from pathlib import Path

from ngio import open_ome_zarr_plate
from ngio.utils import download_ome_zarr_dataset

download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
ome_zarr_plate = open_ome_zarr_plate(hcs_path)
print(ome_zarr_plate)
# --8<-- [end:setup]

# --8<-- [start:plate_columns]
print(ome_zarr_plate.columns)
# --8<-- [end:plate_columns]

# --8<-- [start:plate_rows]
print(ome_zarr_plate.rows)
# --8<-- [end:plate_rows]

# --8<-- [start:plate_acquisitions]
print(ome_zarr_plate.acquisition_ids)
# --8<-- [end:plate_acquisitions]

# --8<-- [start:images_paths]
print(ome_zarr_plate.images_paths())
# --8<-- [end:images_paths]

# --8<-- [start:wells_paths]
print(ome_zarr_plate.wells_paths())
# --8<-- [end:wells_paths]

# --8<-- [start:well_images_paths]
print(ome_zarr_plate.well_images_paths(row="B", column=3))
# --8<-- [end:well_images_paths]

# --8<-- [start:get_images]
print(ome_zarr_plate.get_images(max_workers="auto"))
# --8<-- [end:get_images]

# --8<-- [start:get_well_images]
well_images = ome_zarr_plate.get_well_images(row="B", column=3)
print(well_images)
# --8<-- [end:get_well_images]

# --8<-- [start:get_image]
print(ome_zarr_plate.get_image(row="B", column=3, image_path="0"))
# --8<-- [end:get_image]

# --8<-- [start:get_well_images_by_acquisition]
well_images = ome_zarr_plate.get_well_images(row="B", column=3, acquisition=0)
print(well_images)
# --8<-- [end:get_well_images_by_acquisition]

# --8<-- [start:image_in_well_paths]
from ngio import ImageInWellPath

list_of_images = [
    ImageInWellPath(path="0", row="A", column=0),
    ImageInWellPath(path="0", row="B", column=1),
    ImageInWellPath(path="0", row="C", column=1),
    ImageInWellPath(
        path="1",
        row="A",
        column=0,
        acquisition_id=1,
        acquisition_name="acquisition_1",
    ),
]
# --8<-- [end:image_in_well_paths]

# --8<-- [start:create_empty_plate]
from ngio import create_empty_plate

plate = create_empty_plate(
    store="data/new_plate.zarr",
    name="test_plate",
    images=list_of_images,
    overwrite=True,
)
print(plate)
# --8<-- [end:create_empty_plate]

# --8<-- [start:plate_add_image]
print(f"Before adding images: {plate.rows} rows, {plate.columns} columns")
plate.add_image(row="D", column=0, image_path="0")
print(f"After adding images: {plate.rows} rows, {plate.columns} columns")
# --8<-- [end:plate_add_image]

# --8<-- [start:plate_remove_image]
print(f"Before removing images: {plate.wells_paths()} wells")
plate.remove_image(row="D", column=0, image_path="0")
print(f"After removing images: {plate.wells_paths()} wells")
# --8<-- [end:plate_remove_image]
