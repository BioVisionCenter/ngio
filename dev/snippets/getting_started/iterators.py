"""Snippets for docs/getting_started/6_iterators.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/iterators.py
"""

# --8<-- [start:setup]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.experimental.iterators import ImageProcessingIterator
from ngio.utils import download_ome_zarr_dataset

download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset(
    "CardiomyocyteSmallMip", download_dir=download_dir, re_unzip=False
)
ome_zarr = open_ome_zarr_container(hcs_path / "B" / "03" / "0")
image = ome_zarr.get_image()
# --8<-- [end:setup]

# --8<-- [start:build]
# A new iterator covers the whole image as a single region
iterator = ImageProcessingIterator(input_image=image, output_image=image)
print(iterator)
# --8<-- [end:build]

# --8<-- [start:product]
# Narrow it to the regions named by a ROI table
iterator = iterator.product(ome_zarr.get_roi_table("FOV_ROI_table"))
print(iterator)
# --8<-- [end:product]

# --8<-- [start:inspect]
# The regions are plain Roi objects, so you can look before you process
for roi in iterator.rois[:2]:
    print(roi)
# --8<-- [end:inspect]
