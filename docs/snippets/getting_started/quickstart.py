"""Snippets for docs/getting_started/0_quickstart.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/quickstart.py
"""

# --8<-- [start:setup]
from pathlib import Path

from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"
# --8<-- [end:setup]

# --8<-- [start:open_container]
from ngio import open_ome_zarr_container

ome_zarr_container = open_ome_zarr_container(image_path)
print(ome_zarr_container)
# --8<-- [end:open_container]
