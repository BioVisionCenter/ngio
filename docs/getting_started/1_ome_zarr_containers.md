---
description: "The OME-Zarr container object: inspect and modify metadata, derive and create images, and open remote stores."
---

# 1. OME-Zarr containers

**Open an OME-Zarr image and explore what it holds.**

The OME-Zarr container is your entry point to working with OME-Zarr images. It gives you
high-level access to the metadata, images, labels and tables in a store.

<!-- Figure 01 — the object model -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 412" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f1t f1d">
  <title id="f1t">The objects inside an OME-Zarr container</title>
  <desc id="f1d">An OmeZarrContainer branches into an images container made of one Image per resolution level, a labels container holding named multiscale labels, and a tables container holding typed tables.</desc>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M192 204H208M208 68V340M208 68H216M208 204H216M208 340H216"></path>
    <path d="M424 68H440M440 32V104M440 32H456M440 68H456M440 104H456"></path>
    <path d="M424 204H440M440 168V240M440 168H456M440 204H456M440 240H456"></path>
    <path d="M424 340H440M440 304V376M440 304H456M440 340H456M440 376H456"></path>
  </g>

  <rect x="8" y="176" width="184" height="56" rx="8" style="fill:var(--ngio-surface);stroke:var(--ngio-accent)" stroke-width="1.5"></rect>
  <text x="24" y="200" style="font-family:'JetBrains Mono',monospace;font-size:12.5px;fill:var(--md-default-fg-color)">OmeZarrContainer</text>
  <text x="24" y="219" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">the store, opened</text>

  <rect x="216" y="8" width="416" height="120" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <rect x="216" y="144" width="416" height="120" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <rect x="216" y="280" width="416" height="120" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>

  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="232" y="28">IMAGE · PIXEL DATA</text>
    <text x="232" y="164">LABELS · SEGMENTATIONS</text>
    <text x="232" y="300">TABLES · MEASUREMENTS AND ROIS</text>
  </g>

  <rect x="232" y="36" width="184" height="64" rx="8" style="fill:var(--ngio-surface);stroke:var(--ngio-line-strong)"></rect>
  <g style="fill:var(--ngio-blue)">
    <rect x="246" y="55" width="10" height="7" rx="1.5"></rect>
    <rect x="246" y="64" width="17" height="8" rx="1.5"></rect>
    <rect x="246" y="74" width="26" height="8" rx="1.5"></rect>
  </g>
  <text x="286" y="64" style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">ImagesContainer</text>
  <text x="286" y="82" style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;fill:var(--md-default-fg-color--light)">several resolutions</text>

  <rect x="232" y="172" width="184" height="64" rx="8" style="fill:var(--ngio-surface);stroke:var(--ngio-line-strong)"></rect>
  <rect x="246.5" y="194.5" width="25" height="19" rx="3" style="fill:var(--ngio-sunk);stroke:var(--ngio-line-strong)"></rect>
  <ellipse cx="254" cy="200" rx="3.4" ry="3" style="fill:var(--ngio-green)"></ellipse>
  <ellipse cx="264" cy="207" rx="4.2" ry="3.6" style="fill:var(--ngio-green)"></ellipse>
  <ellipse cx="253" cy="209" rx="2.6" ry="2.2" style="fill:var(--ngio-green)"></ellipse>
  <text x="286" y="200" style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">LabelsContainer</text>
  <text x="286" y="218" style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;fill:var(--md-default-fg-color--light)">masks, by name</text>

  <rect x="232" y="308" width="184" height="64" rx="8" style="fill:var(--ngio-surface);stroke:var(--ngio-line-strong)"></rect>
  <rect x="246.75" y="330.75" width="24.5" height="18.5" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <path d="M246 337h26M255 330v20M264 330v20" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>
  <text x="286" y="336" style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">TablesContainer</text>
  <text x="286" y="354" style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;fill:var(--md-default-fg-color--light)">tables, by name</text>

  <g style="fill:var(--ngio-surface);stroke:var(--ngio-line)">
    <rect x="456" y="18" width="168" height="28" rx="6"></rect>
    <rect x="456" y="54" width="168" height="28" rx="6"></rect>
    <rect x="456" y="90" width="168" height="28" rx="6"></rect>
    <rect x="456" y="154" width="168" height="28" rx="6"></rect>
    <rect x="456" y="190" width="168" height="28" rx="6"></rect>
    <rect x="456" y="290" width="168" height="28" rx="6"></rect>
    <rect x="456" y="326" width="168" height="28" rx="6"></rect>
    <rect x="456" y="362" width="168" height="28" rx="6"></rect>
  </g>
  <rect x="456" y="226" width="168" height="28" rx="6" fill="none" style="stroke:var(--ngio-line-strong)" stroke-dasharray="4 4"></rect>

  <g style="fill:var(--ngio-blue)">
    <rect x="470" y="27" width="10" height="10" rx="2"></rect>
    <rect x="470" y="63" width="10" height="10" rx="2"></rect>
    <rect x="470" y="99" width="10" height="10" rx="2"></rect>
  </g>
  <g style="fill:var(--ngio-green)">
    <rect x="470" y="162" width="4" height="3" rx="1"></rect>
    <rect x="470" y="166" width="7" height="3" rx="1"></rect>
    <rect x="470" y="170" width="11" height="3" rx="1"></rect>
    <rect x="470" y="198" width="4" height="3" rx="1"></rect>
    <rect x="470" y="202" width="7" height="3" rx="1"></rect>
    <rect x="470" y="206" width="11" height="3" rx="1"></rect>
  </g>
  <g style="fill:var(--ngio-magenta)">
    <rect x="470" y="299" width="10" height="10" rx="2"></rect>
    <rect x="470" y="335" width="10" height="10" rx="2"></rect>
    <rect x="470" y="371" width="10" height="10" rx="2"></rect>
  </g>

  <g style="font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:var(--md-default-fg-color)">
    <text x="490" y="36">Image · level 0</text>
    <text x="490" y="72">Image · level 1</text>
    <text x="490" y="108">Image · level 2 …</text>
    <text x="490" y="172">Label · nuclei</text>
    <text x="490" y="208">Label · cells</text>
    <text x="490" y="308">RoiTable</text>
    <text x="490" y="344">FeatureTable</text>
    <text x="490" y="380">MaskingRoiTable</text>
  </g>
  <text x="470" y="244" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">… any number of labels</text>
        </svg>
</div>

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:setup"
```

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:print_container"
```

The OME-Zarr container will be the starting point for all your image processing tasks.

## Main concepts

### What is the OME-Zarr container?

The OME-Zarr container gives you:

- **OME-Zarr overview**: get an overview of the OME-Zarr file, including the number of image levels, list of labels, and tables available.
- **Image access**: get access to the images at different resolution levels and pixel sizes.
- **Label management**: check which labels are available, access them, and create new labels.
- **Table management**: check which tables are available, access them, and create new tables.
- **Derive new OME-Zarr images**: create new images based on the original one, with the same or similar metadata.

### What is the OME-Zarr container not?

The OME-Zarr container does not give you access to the image data directly. For that, use the `Image`, `Label`, and `Table` objects.

## OME-Zarr overview

Examples of accessing the OME-Zarr metadata:

=== "Number of resolution levels"
    Show the number of resolution levels:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:levels"
    ```

=== "Available paths"
    Show the paths to all available resolution levels:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:level_paths"
    ```

=== "Dimensionality"
    Show if the image is 2D or 3D:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:is_3d"
    ```
    or if the image is a time series:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:is_time_series"
    ```

=== "Full metadata object"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:metadata"
    ```
    The metadata object contains all the information about the image, for example, the channel labels:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:channel_labels"
    ```
    And those three channels, read from a lower pyramid level:
    ```python exec="true" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_helpers"
    ```
    ```python exec="true" html="1" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:plot_container_channels"
    ```

## Modifying metadata

ngio provides methods to modify the image metadata, such as channel labels, colours, display windows, axes names, and units.

### Channel metadata

You can update channel labels, colours, and display windows:

=== "Channel labels"
    Update the labels (names) of the channels:
    ```python
    ome_zarr_container.set_channel_labels(["DAPI", "GFP", "RFP"])
    ```

=== "Channel colours"
    Update the display colours of the channels (hex format):
    ```python
    ome_zarr_container.set_channel_colors(["0000FF", "00FF00", "FF0000"])
    ```

=== "Channel windows"
    Update the display windows (start/end values) for each channel:
    ```python
    ome_zarr_container.set_channel_windows([(0, 255), (0, 1000), (0, 500)])
    ```

=== "Channel windows from percentiles"
    Automatically compute display windows based on data percentiles:
    ```python
    ome_zarr_container.set_channel_windows_with_percentiles(percentiles=(0.1, 99.9))
    ```

### Axes metadata

You can update the axes names and units:

=== "Axes names"
    Rename the axes in the metadata:
    ```python
    ome_zarr_container.set_axes_names(["t", "c", "z", "y", "x"])
    ```

=== "Axes units"
    Set the space and time units, each on its own (setting one leaves the
    other untouched; both also apply to the labels unless `set_labels=False`):
    ```python
    ome_zarr_container.set_space_unit("micrometer")
    ome_zarr_container.set_time_unit("second")
    ```

### Image name

You can set the name of the image in the metadata:

```python
ome_zarr_container.set_name("My Processed Image")
```

!!! note
    The `set_name` method only updates the metadata. It does not change the group name or file paths.

## Accessing images / labels / tables

To access images, labels, and tables, you can use the `get_image`, `get_label`, and `get_table` methods of the OME-Zarr container.

A variety of examples and additional information can be found in the [Images and labels](./2_images.md), and [Tables](./3_tables.md) sections.

## Creating derived images

When processing an image, you might want to create a new image with the same metadata:

```python
# Create a new image based on the original
new_image = ome_zarr_container.derive_image("data/new_ome.zarr", overwrite=True)
```

This will create a new OME-Zarr image with the same metadata as the original image.
But you can also create a new image with slightly different metadata, for example, with a different shape:

```python
# Create a new image with a different shape
new_image = ome_zarr_container.derive_image(
    "data/new_ome.zarr",
    overwrite=True,
    shape=(16, 128, 128),
    pixelsize=0.65,
    z_spacing=1.0
)
```

## Creating new images

You can create OME-Zarr images from an existing numpy array using the `create_ome_zarr_from_array` function.

```python
import numpy as np
from ngio import create_ome_zarr_from_array

# Create a random 3D array
x = np.random.randint(0, 255, (16, 128, 128), dtype=np.uint8)

# Save as OME-Zarr
new_ome_zarr_image = create_ome_zarr_from_array(
    store="random_ome.zarr",
    array=x,
    pixelsize=0.65,
    z_spacing=1.0
)
```

Alternatively, if you want to create an empty OME-Zarr image, you can use the `create_empty_ome_zarr` function:

```python
from ngio import create_empty_ome_zarr
# Create an empty OME-Zarr image
new_ome_zarr_image = create_empty_ome_zarr(
    store="empty_ome.zarr",
    shape=(16, 128, 128),
    pixelsize=0.65,
    z_spacing=1.0
)
```

This will create an empty OME-Zarr image with the specified shape and pixel sizes.

### Chunks, shards, and compression

Both creation functions accept the array geometry and codec directly; the settings
apply to **every** pyramid level, clipped per level so a deep level never exceeds
its own shape:

```python
from zarr.codecs import BloscCodec

sharded = create_empty_ome_zarr(
    store="sharded_ome.zarr",
    shape=(16, 2048, 2048),
    pixelsize=0.65,
    ngff_version="0.5",
    chunks=(1, 256, 256),
    shards=(4, 2048, 2048),
    compressors=BloscCodec(cname="zstd", clevel=5),
)
```

A few rules, all enforced before anything is written:

- Sharding needs OME-Zarr 0.5 (zarr format 3); on `ngff_version="0.4"` pass
  `shards=None`, or `shards="auto"`, which means no sharding there.
- An explicit shard shape needs an explicit chunk shape — with `chunks="auto"`
  zarr would infer chunks the shard is not a multiple of.
- On 0.4 the codec is a `numcodecs` object (`numcodecs.Blosc(...)`); on 0.5 a
  zarr v3 codec (`zarr.codecs.BloscCodec(...)`). The two are not interchangeable.

`derive_image` / `derive_label` inherit `chunks`, `shards`, `compressors`,
`dtype`, and the dimension separator from the reference image unless you pass a
value explicitly. When deriving *across* formats (`ngff_version=` differs from
the reference), inheriting codecs or shards cannot work, so ngio asks you to be
explicit: `compressors="auto"` selects the target format's default, and
`shards="auto"` derives unsharded onto a 0.4 target.

Anything beyond these (`fill_value`, `filters`, `serializer`, ...) goes through
`extra_array_kwargs`, which is forwarded to `zarr.create_array` as-is.

## Opening remote OME-Zarr containers

You can use `ngio` to open remote OME-Zarr containers.
For publicly available OME-Zarr containers, use the `open_ome_zarr_container` function with a URL.

For example, to open a remote OME-Zarr container hosted on a GitHub repository:

```python
from ngio import open_ome_zarr_container
from ngio.utils import fractal_fsspec_store

url = (
    "https://raw.githubusercontent.com/"
    "fractal-analytics-platform/fractal-ome-zarr-examples/"
    "refs/heads/main/v04/"
    "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
)

store = fractal_fsspec_store(url=url)
ome_zarr_container = open_ome_zarr_container(store)
```

For Fractal users, the `fractal_fsspec_store` function can be used to open private OME-Zarr containers.
In this case you need to provide a `fractal_token` to authenticate.

```python
from ngio import open_ome_zarr_container
from ngio.utils import fractal_fsspec_store

store = fractal_fsspec_store(url="https://fractal_url...", fractal_token="**your_secret_token**")
ome_zarr_container = open_ome_zarr_container(store)
```

## Caching and freshness

`open_ome_zarr_container(store, cache=...)` decides how long metadata is
trusted:

- **`cache=False` (the default)** re-reads the raw metadata from the store on
  every access, so writes made through *another* handle — another process, a
  Fractal task, a second container object — are picked up as they happen.
- **`cache=True`** holds all metadata for the object's lifetime: after the
  first read, listing tables or reading pixel sizes costs no store requests.
  The trade is staleness — the object will not see outside writes until you
  ask.

`refresh()` is the escape hatch for both modes: it re-reads everything the
container holds. It is not a no-op even under `cache=False` — a few derived
values (an image's decoded metadata and its `dimensions`) are memoized per
object regardless of the flag, and `refresh()` drops those too.

Writes made through the *same* container are always visible to it,
whatever the mode.

## Next steps

- [Images and labels](2_images.md) — read and write pixel data.
- [Tables](3_tables.md) — ROIs, features and measurements stored alongside the image.
