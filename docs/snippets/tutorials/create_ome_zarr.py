"""Snippets for docs/tutorials/create_ome_zarr.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/create_ome_zarr.py
"""

# --8<-- [start:plot_helpers]
from io import StringIO

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")


def print_figure(fig: Figure) -> None:
    """Print a figure as inline SVG, for markdown-exec `html` blocks."""
    buffer = StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    print(buffer.getvalue())


# --8<-- [end:plot_helpers]

# --8<-- [start:plot_input_image]
import skimage

fig, ax = plt.subplots()
ax.imshow(skimage.data.human_mitosis(), cmap="gray")
ax.axis("off")
print_figure(fig)
# --8<-- [end:plot_input_image]

# --8<-- [start:create]
from ngio import create_ome_zarr_from_array

ome_zarr = create_ome_zarr_from_array(
    store="./data/human_mitosis.zarr",
    array=skimage.data.human_mitosis(),
    pixelsize=0.1,  # Just a guess
    overwrite=True,
)
print(ome_zarr)
# --8<-- [end:create]

# --8<-- [start:add_roi_table]
# create a roi for the whole image
roi_table = ome_zarr.build_image_roi_table(name="image_roi")
ome_zarr.add_table("image_roi_table", roi_table, overwrite=True)
# --8<-- [end:add_roi_table]
