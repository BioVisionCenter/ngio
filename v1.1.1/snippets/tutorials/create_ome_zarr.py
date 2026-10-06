"""Snippets for docs/tutorials/create_ome_zarr.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/create_ome_zarr.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html
# --8<-- [end:plot_helpers]

# --8<-- [start:plot_input_image]
import skimage

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(skimage.data.human_mitosis(), cmap="gray")
ax.axis("off")
print(figure_html(fig))
# --8<-- [end:plot_input_image]

# --8<-- [start:create]
from ngio import create_ome_zarr_from_array

ome_zarr = create_ome_zarr_from_array(
    store="./data/human_mitosis.zarr",
    array=skimage.data.human_mitosis(),
    pixelsize=0.1,  # Just a guess
    consolidation_mode="auto",
    overwrite=True,
)
print(ome_zarr)
# --8<-- [end:create]

# --8<-- [start:add_roi_table]
# create a roi for the whole image
roi_table = ome_zarr.build_image_roi_table(name="image_roi")
ome_zarr.add_table("image_roi_table", roi_table, overwrite=True)
# --8<-- [end:add_roi_table]
