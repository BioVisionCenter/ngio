---
description: "Read and write OME-Zarr pixel data: resolution levels, numpy and dask access, slicing, labels, merge policies, and region-scoped consolidation."
---

# 2. Images and labels

**Read and write the pixel data.**

An [`Image`][ngio.Image] gives you the data of one resolution level: as numpy or dask, sliced
by axis or by a region of interest in world coordinates. A [`Label`][ngio.Label] is a
segmentation stored the same way, and behaves the same way.

<!-- Figure 02 — resolution levels -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 296" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f2t f2d">
  <title id="f2t">A multiscale pyramid</title>
  <desc id="f2d">Every level covers the same physical region of the sample. Each step doubles the pixel size, so the same area is stored with a quarter of the pixels.</desc>

  <defs>
    <filter id="n2g" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch"></feTurbulence><feColorMatrix type="saturate" values="0"></feColorMatrix></filter>
    <filter id="n2b1" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.2"></feGaussianBlur></filter>
    <filter id="n2b2" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2.8"></feGaussianBlur></filter>
    <g id="n2cells" fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.7" stroke-opacity=".92">
      <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path><g fill="#ffffff" stroke="none"><circle cx="-9" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-1" cy="3" r="1" opacity="0.7"></circle><circle cx="7" cy="-2.5" r="1.1" opacity="0.75"></circle><circle cx="13" cy="1.5" r="0.8" opacity="0.6"></circle></g></g>
      <g transform="translate(55,34) rotate(6)"><path d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-4" cy="-3" r="1.2" opacity="0.8"></circle><circle cx="3" cy="2" r="1" opacity="0.65"></circle><circle cx="6" cy="-3" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(78,63) rotate(-34)"><path d="M-12-1.4c.6-3.2 5-4.8 10.6-4.6 6.4.2 12.4 2 13.8 4.8 1 2-1.4 5.2-6.4 6.8-5.6 1.8-12 1.4-15.6-1-1.8-1.2-2.6-3.6-2.4-6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-1" r="1.1" opacity="0.75"></circle><circle cx="2" cy="1.5" r="0.9" opacity="0.6"></circle><circle cx="8" cy="-1" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-11" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-3" cy="3" r="1.1" opacity="0.7"></circle><circle cx="6" cy="-3" r="1.2" opacity="0.75"></circle><circle cx="14" cy="2" r="0.9" opacity="0.6"></circle><circle cx="1" cy="-6" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(20,60) rotate(62)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-2" r="1.2" opacity="0.8"></circle><circle cx="2" cy="2.5" r="1" opacity="0.65"></circle><circle cx="8" cy="-1.5" r="0.9" opacity="0.6"></circle></g></g>
      <g transform="translate(84,20) rotate(-8)"><path d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path><g fill="#ffffff" stroke="none"><circle cx="-2" cy="-2" r="1" opacity="0.7"></circle><circle cx="3" cy="2" r="0.9" opacity="0.6"></circle></g></g>
    </g>
    <g id="n2speck" fill="#ffffff">
      <circle cx="12" cy="14" r=".7" opacity=".4"></circle><circle cx="38" cy="10" r=".5" opacity=".28"></circle><circle cx="56" cy="34" r=".8" opacity=".32"></circle><circle cx="18" cy="44" r=".6" opacity=".22"></circle><circle cx="72" cy="42" r=".5" opacity=".3"></circle><circle cx="88" cy="24" r=".7" opacity=".24"></circle><circle cx="34" cy="66" r=".6" opacity=".28"></circle><circle cx="52" cy="72" r=".5" opacity=".2"></circle><circle cx="78" cy="70" r=".8" opacity=".3"></circle><circle cx="14" cy="58" r=".5" opacity=".18"></circle><circle cx="92" cy="86" r=".6" opacity=".26"></circle><circle cx="44" cy="92" r=".7" opacity=".22"></circle><circle cx="66" cy="58" r=".5" opacity=".2"></circle><circle cx="24" cy="90" r=".6" opacity=".28"></circle>
    </g>
    <clipPath id="n2p0"><rect x="32" y="62" width="144" height="144" rx="3"></rect></clipPath>
    <clipPath id="n2p1"><rect x="248" y="62" width="144" height="144" rx="3"></rect></clipPath>
    <clipPath id="n2p2"><rect x="464" y="62" width="144" height="144" rx="3"></rect></clipPath>
  </defs>

  <g style="fill:var(--ngio-sunk);stroke:var(--ngio-line)">
    <rect x="32.5" y="8.5" width="143" height="25" rx="6"></rect>
    <rect x="248.5" y="8.5" width="143" height="25" rx="6"></rect>
    <rect x="464.5" y="8.5" width="143" height="25" rx="6"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:12.5px;fill:var(--md-default-fg-color)">
    <text x="104" y="26">level 0</text>
    <text x="320" y="26">level 1</text>
    <text x="536" y="26">level 2</text>
  </g>
  <g text-anchor="middle" style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">
    <text x="104" y="52">full resolution</text>
    <text x="320" y="52">half in each axis</text>
    <text x="536" y="52">a quarter in each axis</text>
  </g>

  <g clip-path="url(#n2p0)">
    <rect x="32" y="62" width="144" height="144" fill="#151d21"></rect>
    <use href="#n2cells" transform="translate(32,62) scale(1.44)"></use>
    <use href="#n2speck" transform="translate(32,62) scale(1.44)"></use>
    <rect x="32" y="62" width="144" height="144" filter="url(#n2g)" opacity=".10" style="mix-blend-mode:screen"></rect>
  </g>
  <path d="M50 62V206M68 62V206M86 62V206M104 62V206M122 62V206M140 62V206M158 62V206M32 80H176M32 98H176M32 116H176M32 134H176M32 152H176M32 170H176M32 188H176" stroke="#ffffff" stroke-width="1.1" opacity=".28"></path>

  <g clip-path="url(#n2p1)">
    <rect x="248" y="62" width="144" height="144" fill="#151d21"></rect>
    <g filter="url(#n2b1)"><use href="#n2cells" transform="translate(248,62) scale(1.44)"></use></g>
    <rect x="248" y="62" width="144" height="144" filter="url(#n2g)" opacity=".07" style="mix-blend-mode:screen"></rect>
  </g>
  <path d="M284 62V206M320 62V206M356 62V206M248 98H392M248 134H392M248 170H392" stroke="#ffffff" stroke-width="1.4" opacity=".32"></path>

  <g clip-path="url(#n2p2)">
    <rect x="464" y="62" width="144" height="144" fill="#151d21"></rect>
    <g filter="url(#n2b2)"><use href="#n2cells" transform="translate(464,62) scale(1.44)"></use></g>
  </g>
  <path d="M536 62V206M464 134H608" stroke="#ffffff" stroke-width="1.4" opacity=".36"></path>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M32 216v6M32 219h144M176 216v6"></path>
    <path d="M248 216v6M248 219h144M392 216v6"></path>
    <path d="M464 216v6M464 219h144M608 216v6"></path>
  </g>

  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">
    <text x="104" y="244">4320 × 5120 px</text>
    <text x="320" y="244">2160 × 2560 px</text>
    <text x="536" y="244">1080 × 1280 px</text>
  </g>
  <g text-anchor="middle" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">
    <text x="104" y="262">0.325 µm per pixel</text>
    <text x="320" y="262">0.65 µm per pixel</text>
    <text x="536" y="262">1.3 µm per pixel</text>
  </g>
  <text x="320" y="288" text-anchor="middle" style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color--light)">the physical region never changes — only how finely it is sampled</text>
        </svg>
</div>

## Images

To start working with the image data, instantiate an [`Image`][ngio.Image] object.
ngio provides a high-level API to access the image data at different resolution levels and pixel sizes.

### Getting an image

```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:reopen_container"
```
```python exec="true" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:plot_helpers"
```

=== "Highest resolution image"
    By default, the `get_image` method returns the highest resolution image:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_default"
    ```

=== "Specific pyramid level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_by_path"
    ```
    This will return the image at the specified pyramid level.

=== "Specific resolution"
    If you want to get an image with a specific pixel size, you can use the `pixel_size` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_by_pixel_size"
    ```

=== "Nearest resolution"
    ngio returns the level whose pixel size is nearest to the one you ask for. That is the default, `strict=False`, spelled out here:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_nearest"
    ```
    Pass `strict=True` instead to require an exact match, and raise `NgioValueError` when no level has that pixel size. The module-level [`open_image`][ngio.open_image] and [`open_label`][ngio.open_label] functions default the other way, to `strict=True`.

Similarly to the OME-Zarr container, the `Image` object provides a high-level API to access the image metadata.

=== "Dimensions"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_dimensions"
    ```
    The `dimensions` attribute returns an object with the image dimensions for each axis.

=== "Pixel size"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_pixel_size"
    ```
    The `pixel_size` attribute returns the pixel size for each axis.

=== "On-disk array info"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_array_info"
    ```
    `shape`, `dtype` and `chunks` come straight from the underlying Zarr array; `axes` gives the order those dimensions are stored in.

### Working with image data

Once you have the `Image` object, you can access the image data as a:

=== "Numpy array"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_as_numpy"
    ```

=== "Dask array"
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_as_dask"
    ```

=== "Either, by mode"
    `get_array` is the generic form of the two above: one entry point that picks the backend from a `mode` argument. Reach for it when the backend is decided at runtime; otherwise prefer the explicit `get_as_numpy` / `get_as_dask`.

    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_get_array_legacy"
    ```

The `get_as_*` methods can also slice the image data, and return the axes in an order you choose:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:image_slice"
```

To write pixel data back, use the `set_array` method:

```python
image.set_array(data)
```

It accepts a numpy array or a dask array, and takes the same slicing and `axes_order`
arguments as the getters, so you can write back exactly the region you read. Writes
can also *combine* with what is on disk instead of replacing it — see
[Merging instead of overwriting](#merging-instead-of-overwriting) and
[Keeping label ids unique across regions](#keeping-label-ids-unique-across-regions)
below.

A minimal read-modify-write example:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:set_array_example"
```

!!! important
    `set_array` writes to one resolution level only. Once you have finished editing, consolidate the changes so the rest of the pyramid is rebuilt from it:
    ```python
    image.consolidate()
    ```
    If your edits touch only part of a large image, record them with `track_writes` and consolidate selectively — only the pyramid regions that derive from the writes are rebuilt, with results identical to a full rebuild:
    ```python
    with image.track_writes() as regions:
        image.set_roi(roi, patch)
        image.set_array(other, y=slice(0, 64), x=slice(0, 64))
    image.consolidate(regions=regions)
    ```
    Only `set_*` calls made through this image handle are recorded — writes through other handles, custom setter pipes, or worker processes are not seen.

### World coordinates slicing

To read or write a specific region of the image defined in world coordinates, you can use the [`Roi`][ngio.Roi] object.

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:roi_slicing"
```

The ROI is defined in micrometres, so it names the same region whatever pyramid level you
read it from — on the left it is outlined on the whole image, on the right it is the region
that came back:

```python exec="true" html="1" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:plot_roi_slicing"
```

## Labels

A label is a segmentation mask that identifies objects in the image. In ngio a [`Label`][ngio.Label]
behaves like an [`Image`][ngio.Image], and is accessed and manipulated the same way.

### Getting a label

See which labels are available in the image:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:list_labels"
```

Here is how to reach one of them:

=== "Highest resolution label"
    By default, the `get_label` method returns the highest resolution label:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_default"
    ```

=== "Specific pyramid level"
    To get a specific pyramid level, you can use the `path` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_by_path"
    ```
    This will return the label at the specified pyramid level.

=== "Specific resolution"
    If you want to get a label with a specific pixel size, you can use the `pixel_size` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_by_pixel_size"
    ```

=== "Nearest resolution"
    As with images, the nearest level wins unless you ask for an exact match with `strict=True`:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_nearest"
    ```

Each object in a label carries its own id, drawn here in its own colour over the channel it
was segmented from:

```python exec="true" html="1" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:plot_label_overlay"
```

### Working with label data

Reading and writing label data works exactly as it does for images: `get_as_numpy`, `get_as_dask`, `get_roi_as_numpy` and `set_array` are all available on a `Label`.

### Deriving a label

Often, you might want to create a new label based on an existing image. You can do this using the `derive_label` method:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:derive_label"
```

This will create a new label with the same dimensions as the original image (without channels) and compatible metadata.
If you want to create a new label with slightly different metadata see the [images API reference](../api/images.md).

## Merging instead of overwriting

By default a write replaces what is on disk. `merge=` combines with it instead:

```python
image.set_roi(roi, patch, merge="max")
```

`"max"`, `"min"` and `"sum"` are commutative and associative, so overlapping regions give the same answer whatever order they are written in. `"keep_nonzero"` ("the last nonzero write wins") and a custom `(existing, patch, ctx) -> array` rule do depend on the order.

The merge is a separate argument rather than an entry in `transforms=`: a transform is a function of the patch alone, while a merge also depends on what is already there — so it runs once, after the chain, with both sides in the array's own space. That is what makes the comparison meaningful and keeps untouched pixels byte-identical.

Masking follows the same split. On a read it fills outside the mask, which is a transform; on a write it protects outside the mask, which is a merge:

```python
from ngio.transforms import MaskMerge, MaskTransform

patch = image.get_roi(roi, transforms=[MaskTransform(label=nuclei, target_image=image)])
image.set_roi(roi, patch, merge=MaskMerge(label=nuclei, target_image=image))
```

`get_roi_masked` and `set_roi_masked` do this for you; reach for the objects directly when you want to combine masking with other transforms.

## Keeping label ids unique across regions

Segmenting region by region gives each region its own `1, 2, 3, …`, so writing them into one array collides. `UniqueLabelsTransform` gives each region a disjoint slice of the id space:

```python
from ngio.transforms import UniqueLabelsTransform

# Region 4's labels 1, 2, 3 are written as 4001, 4002, 4003.
label.set_roi(roi, patch, transforms=[UniqueLabelsTransform(1000, block_index=4)])
```

`block_size` has to exceed the largest label any one region can produce, or ids spill into the next region's block. Inside a masked iterator you can leave `block_index` out — the ROI's own label supplies it.

The offset is derived from the block index rather than counted up as regions are processed, so it is parallel-safe (no shared counter to synchronize), survives `ProcessMapper`, and is idempotent — a re-run region reproduces exactly the ids it wrote before.

Being an ordinary transform, it composes with a merge:

```python
label.set_roi(
    roi, patch, transforms=[UniqueLabelsTransform(1000, block_index=4)], merge="max"
)
```

## Next steps

- [Tables](3_tables.md) — use ROIs to slice the image data you just learned to read.
- [Masked images and labels](4_masked_images.md) — work object-by-object using a segmentation.
- [Images API reference](../api/images.md) — every method on `Image` and `Label`.
