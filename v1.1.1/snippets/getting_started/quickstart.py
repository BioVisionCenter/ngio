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

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, show_image
# --8<-- [end:plot_helpers]

# --8<-- [start:plot_quickstart_image]
image = ome_zarr_container.get_image(path="3")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
show_image(
    ax,
    image.get_as_numpy(channel_selection="DAPI"),
    title="DAPI · level 3",
    pixel_size=image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="A field of cardiomyocyte nuclei, stained with DAPI."))
# --8<-- [end:plot_quickstart_image]
