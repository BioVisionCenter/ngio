"""Rendering helpers shared by every executed docs snippet.

Figures and tables reach the site through this module: `figure_html` and `table_html`
build the markup, and the rcParams applied at import give every figure one house style.
Snippet scripts pull it in from their `plot_helpers` / `table_helpers` sections, which
the pages include hidden (`exec="true"` with no `source=`), so none of this plumbing is
reader-facing — but `docs/tutorials/*.md` do show their own plotting code, so keep what
is used there to `figure_html` alone and leave plain matplotlib in view.

Nothing here prints, and that is load-bearing rather than a style choice. markdown-exec
does not redirect `sys.stdout`; it injects its own `print` into the globals of the code
block it executes. A `print` inside this module resolves to the builtin instead, so its
output would land on the build's terminal and the block would render as empty, silently:
the build still exits 0. Hence `print(figure_html(fig))` at every call site: the `print`
has to happen in the block.

Zensical ignores `exclude_docs`, so this file is copied into the built site as a static
asset at `site/snippets/_render.py`, like the snippet scripts themselves. Harmless: a
`.py` does not become a page.
"""

import math
from html import escape
from io import StringIO
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

if TYPE_CHECKING:
    import pandas as pd

    from ngio import PixelSize, Roi

matplotlib.use("Agg")

# Every colour below is the light-scheme value of the matching custom property in
# docs/stylesheets/ngio.css, and doubles as the sentinel `print_figure` swaps for that
# property. A figure therefore degrades to the documented light-mode colour if the swap
# ever stops firing, rather than to black.
INK = "#5b6569"  # --md-default-fg-color--light: titles, ticks, spines
BLUE = "#2e6fd6"  # --ngio-blue: image data
GREEN = "#4cae4f"  # --ngio-green: labels
MAGENTA = "#c2185b"  # --ngio-magenta: tables and ROIs
ACCENT = "#22a699"  # --ngio-accent: ngio itself

# Scale bars are drawn over pixels, so they are fixed white with a dark halo: they have
# to survive any pixel value, not any page colour. Deliberately not a token, and
# deliberately absent from _THEME_VARS below.
ON_IMAGE = "#ffffff"

_THEME_VARS = {
    INK: "var(--md-default-fg-color--light)",
    BLUE: "var(--ngio-blue)",
    GREEN: "var(--ngio-green)",
    MAGENTA: "var(--ngio-magenta)",
    ACCENT: "var(--ngio-accent)",
}

# The house style. Applied once at import, so it reaches figures built by reader-facing
# code that calls nothing from this module.
#
# `figure.figsize` is a starting point, not a house rule. Figures carry their intrinsic
# size into the page (the stylesheet only caps them at the column width) and
# `savefig.bbox: "tight"` trims the canvas back to the panels, so a figure showing a
# whole image should be sized to fill the ~8.6in content column — otherwise it lands at
# whatever width its aspect ratio leaves. A single object cropped to its ROI wants the
# opposite: small, because there is nothing there to enlarge.
#
# `svg.fonttype: "none"` keeps labels as `<text>` rather than outlined paths, so the
# inline SVG resolves the site's own webfont and figure labels match the hand-authored
# diagrams. Text metrics still come from the local fallback, so nothing here may depend
# on exact text width — titles stay single-line and left-aligned, and the scale-bar
# label is anchored at its left edge.
HOUSE_STYLE: dict[str, Any] = {
    "figure.figsize": (8.0, 4.0),
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "svg.fonttype": "none",
    "font.family": "monospace",
    "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono"],
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    # "medium" would print a `findfont: Failed to find font weight` line on every build.
    "axes.titleweight": "normal",
    "axes.titlepad": 5.0,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.4,
}

plt.style.use(HOUSE_STYLE)


def random_label_cmap(n_labels: int = 1000, seed: int = 0) -> ListedColormap:
    """Build a reproducible random colormap for label images."""
    rng = np.random.default_rng(seed)
    colors = rng.random((n_labels, 3))
    colors[0] = 0.0
    return ListedColormap(colors)


def stretch_limits(
    data: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.8),
    ignore_zeros: bool = False,
) -> tuple[float, float]:
    """Return the display window for an intensity image.

    Microscopy data rarely fills its dtype — a uint16 MIP shown on the full 0-65535
    range is nearly black — so figures window on percentiles instead.

    Args:
        data: The intensity array.
        percentiles: Lower and upper percentile bounding the window.
        ignore_zeros: Compute the window over non-zero values only. For masked data,
            where the zeros outside the mask would otherwise dominate.

    Returns:
        The `(vmin, vmax)` pair, widened to a unit range if the data is constant.
    """
    values = np.asarray(data)
    if ignore_zeros:
        non_zero = values[values > 0]
        if non_zero.size:
            values = non_zero
    vmin, vmax = np.percentile(values, percentiles)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def show_image(
    ax: Axes,
    data: np.ndarray,
    *,
    title: str | None = None,
    cmap: str | Colormap = "gray",
    limits: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = (1.0, 99.8),
    ignore_zeros: bool = False,
    alpha: float | None = None,
    mask_zeros: bool = False,
    pixel_size: "PixelSize | None" = None,
) -> AxesImage:
    """Draw one image panel in the house style.

    Args:
        ax: The axes to draw on.
        data: The array to show; squeezed first, so singleton `c`/`z`/`t` axes are fine.
        title: Panel title.
        cmap: A colormap name for intensity data, or a `Colormap` for labels — a
            `Colormap` also turns off the intensity window and any interpolation.
        limits: An explicit `(vmin, vmax)`, bypassing `stretch_limits`. Pass the same
            pair to both panels of a before/after figure: two independently stretched
            panels would misrepresent the change between them.
        percentiles: Forwarded to `stretch_limits`.
        ignore_zeros: Forwarded to `stretch_limits`.
        alpha: Opacity, for drawing an overlay over an earlier panel.
        mask_zeros: Hide zero-valued pixels. For a label overlay, so the image below
            shows through at full contrast instead of being dimmed by the background.
        pixel_size: Draw a scale bar from this pixel size.

    Returns:
        The `AxesImage`, so a caller can add a colorbar or overlay another array.
    """
    array = np.squeeze(np.asarray(data))
    if mask_zeros:
        array = np.ma.masked_where(array == 0, array)

    kwargs: dict[str, Any] = {"cmap": cmap}
    if isinstance(cmap, str):
        vmin, vmax = (
            limits
            if limits is not None
            else stretch_limits(array, percentiles, ignore_zeros)
        )
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    else:
        # A label colormap indexes into its own colours: windowing would remap the ids,
        # and smoothing would blend them into colours no object has.
        kwargs["interpolation"] = "nearest"
    if alpha is not None:
        kwargs["alpha"] = alpha

    mappable = ax.imshow(array, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    if pixel_size is not None:
        add_scale_bar(ax, pixel_size)
    return mappable


def _nice_length(target: float) -> float:
    """Round a length to the nearest 1, 2 or 5 per decade."""
    if target <= 0:
        return 1.0
    candidates = [m * 10.0**e for e in range(-4, 8) for m in (1.0, 2.0, 5.0)]
    return min(candidates, key=lambda c: abs(math.log10(c / target)))


def add_scale_bar(ax: Axes, pixel_size: "PixelSize", *, fraction: float = 0.22) -> None:
    """Draw a scale bar in the lower right of an image panel.

    The bar length is the round number nearest `fraction` of the panel width, so it
    stays legible whatever the crop. Both artists are placed in axes coordinates: the
    y axis of an `imshow` is inverted, and axes fractions are not.

    Args:
        ax: The axes holding the image.
        pixel_size: The pixel size the image was read at.
        fraction: Target bar length as a fraction of the panel width.
    """
    x_low, x_high = sorted(ax.get_xlim())
    width_px = abs(x_high - x_low)
    if not width_px:
        return

    length = _nice_length(width_px * fraction * pixel_size.x)
    length_frac = min(length / pixel_size.x / width_px, 0.8)

    unit = pixel_size.space_unit
    symbol = "µm" if unit == "micrometer" else str(unit) if unit else "px"
    halo = [withStroke(linewidth=1.6, foreground="#000000")]

    right, bottom = 0.96, 0.055
    ax.add_patch(
        Rectangle(
            (right - length_frac, bottom),
            length_frac,
            0.018,
            transform=ax.transAxes,
            facecolor=ON_IMAGE,
            edgecolor="none",
            path_effects=halo,
            zorder=5,
        )
    )
    ax.text(
        right - length_frac,
        bottom + 0.055,
        f"{length:g} {symbol}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=ON_IMAGE,
        path_effects=halo,
        zorder=6,
    )


def add_roi_rectangle(
    ax: Axes,
    roi: "Roi",
    pixel_size: "PixelSize",
    *,
    color: str = MAGENTA,
    lw: float = 1.6,
) -> Rectangle:
    """Outline a ROI on an image panel.

    Takes the ROI in world coordinates and converts it, so the caller does not repeat
    the `to_pixel` arithmetic.

    Args:
        ax: The axes holding the image.
        roi: The ROI, in world coordinates.
        pixel_size: The pixel size the image was read at.
        color: Outline colour. Magenta, the docs' semantic colour for tables and ROIs.
        lw: Outline width.

    Returns:
        The rectangle that was added.

    Raises:
        ValueError: If the ROI is unbounded in x or y, and so has no rectangle to draw.
    """
    pixel_roi = roi.to_pixel(pixel_size=pixel_size)
    x_slice = pixel_roi.get("x")
    y_slice = pixel_roi.get("y")
    # A `RoiSlice` bound may be None, meaning "to the edge of the image". Nothing here
    # knows where that edge is, so refuse rather than guess.
    if x_slice is None or y_slice is None:
        raise ValueError(f"ROI {roi.name!r} has no x/y extent to outline.")
    x_start, y_start = x_slice.start, y_slice.start
    width, height = x_slice.length, y_slice.length
    if x_start is None or y_start is None or width is None or height is None:
        raise ValueError(f"ROI {roi.name!r} is unbounded in x or y.")
    rectangle = Rectangle(
        (x_start, y_start),
        width,
        height,
        edgecolor=color,
        facecolor="none",
        lw=lw,
    )
    ax.add_patch(rectangle)
    return rectangle


def figure_html(fig: Figure, alt: str | None = None) -> str:
    """Render a figure as inline SVG, for markdown-exec `html` blocks.

    Swaps the brand colours the figure was drawn with for the matching theme variables,
    so the figure follows the light/dark toggle rather than baking one scheme into the
    page. This only works because the SVG is inline: an `<img src>` would be a separate
    document and would not see the site's custom properties. Colours inside a raster —
    a label overlay, the pixels themselves — are baked into a base64 PNG and cannot
    follow the toggle, which is why greyscale is the rule for pixel data.

    The `.ngio-figure` wrapper is what the stylesheet keys on to strip the `OUT`
    terminal-output treatment that `.result` applies by default.

    Args:
        fig: The figure to render. Closed before returning.
        alt: A short description of what the figure shows, for screen readers.

    Returns:
        The markup to print from the code block.
    """
    buffer = StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)

    # Drop matplotlib's XML declaration and DOCTYPE — they are meaningless inside an
    # HTML body, and the figure is being embedded, not served.
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    for sentinel, css_var in _THEME_VARS.items():
        svg = svg.replace(sentinel, css_var)
    if alt is not None:
        svg = svg.replace("<svg ", f'<svg role="img" aria-label="{escape(alt)}" ', 1)
    return f'<div class="ngio-figure">{svg}</div>'


def table_html(df: "pd.DataFrame") -> str:
    """Render a DataFrame as HTML that the docs theme will style.

    Markdown is not an option here: Zensical does not run block-level Markdown over
    markdown-exec output, so a pipe table would stay literal text. The theme styles
    only `table:not([class])` — and its JS only wraps such tables in a horizontal
    scroll container — while pandas tags its output `class="dataframe"`, so the class
    and the presentational border are stripped.

    Returns:
        The markup to print from the code block.
    """
    # A named index (here the label id) is real data, so promote it to a column: pandas
    # otherwise renders it as a second, near-empty header row.
    if df.index.name is not None:
        df = df.reset_index()
    html = df.to_html(index=False, border=0, float_format="{:.2f}".format)
    return html.replace(' class="dataframe"', "")
