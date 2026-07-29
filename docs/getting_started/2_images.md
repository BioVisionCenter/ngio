---
description: "Read and write OME-Zarr pixel data: resolution levels, numpy and dask access, slicing, and labels."
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
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_image_nearest"
    ```
    This will return the image with the nearest resolution to the requested pixel size.

Similarly to the `OME-Zarr Container`, the `Image` object provides a high-level API to access the image metadata.

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
    The `axes` attribute returns the order of the axes in the image.

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

=== "Legacy"
    A generic `get_array` method is still available for backwards compatibility.

    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:image_get_array_legacy"
    ```

The `get_as_*` can also be used to slice the image data, and query specific axes in specific orders:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:image_slice"
```

If you want to edit the image data, you can use the `set_array` method:

```python
>>> image.set_array(data) # Set the image data
```

The `set_array` method can be used to set the image data from a numpy array, dask array, or dask delayed object.

A minimal example of how to use the `get_array` and `set_array` methods:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:set_array_example"
```

!!! important
    The `set_array` method will overwrite the image data at single resolution level. After you have finished editing the image data, you need to `consolidate` the changes to the OME-Zarr file at all resolution levels:
    ```python
    >>> image.consolidate() # Consolidate the changes
    ```
    This will write the changes to the OME-Zarr file at all resolution levels.

### World coordinates slicing

To read or write a specific region of the image defined in world coordinates, you can use the [`Roi`][ngio.Roi] object.

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:roi_slicing"
```

## Labels

`Labels` represent segmentation masks that identify objects in the image. In ngio `Labels` are similar to `Images` and can
be accessed and manipulated in the same way.

### Getting a label

See which labels are available in the image:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:list_labels"
```

There are `4` labels available in this image. Here is how to access them:

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
    By default the pixels must match exactly the requested pixel size. If you want to get the nearest resolution, you can use the `strict` parameter:
    ```python exec="true" source="material-block" session="get_started"
    --8<-- "docs/snippets/getting_started/get_started.py:get_label_nearest"
    ```
    This will return the label with the nearest resolution to the requested pixel size.

### Working with label data

Data access and manipulation for `Labels` is similar to `Images`. You can use the `get_array` and `set_array` methods to access and modify the label data.

### Deriving a label

Often, you might want to create a new label based on an existing image. You can do this using the `derive_label` method:

```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:derive_label"
```

This will create a new label with the same dimensions as the original image (without channels) and compatible metadata.
If you want to create a new label with slightly different metadata see the [images API reference](../api/images.md).

## Next steps

- [Tables](3_tables.md) — use ROIs to slice the image data you just learned to read.
- [Masked images and labels](4_masked_images.md) — work object-by-object using a segmentation.
- [Images API reference](../api/images.md) — every method on `Image` and `Label`.
