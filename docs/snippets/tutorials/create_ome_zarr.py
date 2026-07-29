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
    """Print a figure as inline SVG, for markdown-exec `html` blocks.

    Recolours every bit of chrome to one sentinel, then swaps that sentinel for
    a theme variable in the emitted markup. The figure therefore follows the
    light/dark toggle rather than baking black text on white into the page.
    This only works because the SVG is inline: an `<img src>` would be a
    separate document and would not see the site's custom properties.

    The `.ngio-figure` wrapper is what the stylesheet keys on to strip the
    `OUT` terminal-output treatment that `.result` applies by default.
    """
    ink = "#5b6569"
    for ax in fig.axes:
        ax.tick_params(colors=ink, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(ink)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(ink)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(ink)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color(ink)
    for text in fig.texts:
        text.set_color(ink)

    buffer = StringIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)

    # Drop matplotlib's XML declaration and DOCTYPE — they are meaningless
    # inside an HTML body, and the figure is being embedded, not served.
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = svg.replace(ink, "var(--md-default-fg-color--light)")
    print(f'<div class="ngio-figure">{svg}</div>')


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
