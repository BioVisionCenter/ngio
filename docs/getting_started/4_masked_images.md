---
description: Read and write image data object-by-object using a segmentation as a mask.
---

# 4. Masked images and labels

**Read and write image data one object at a time.**

A masked image is an image paired with an instance segmentation. Instead of slicing by
coordinates, you address the data by label id, and you can restrict reads and writes to
the pixels belonging to that object.

<!-- Figure 03 — one object at a time -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 240" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f3t f3d">
  <title id="f3t">From a label image to a single masked object</title>
  <desc id="f3d">Each object in a label image gets one row in a masking ROI table. Selecting a row returns just that object's bounding box, with the pixels outside the object masked out.</desc>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="16" y="16">LABEL IMAGE</text>
    <text x="274" y="16">MASKING ROI TABLE</text>
    <text x="492" y="16">ONE OBJECT</text>
  </g>

  <rect x="16.5" y="28.5" width="199" height="167" rx="3" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <g transform="translate(16,28) scale(2,1.68)">
    <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z" fill="#4cae4f"></path></g>
    <g transform="translate(55,34) rotate(6)"><path d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z" fill="#7c6bd6"></path></g>
    <g transform="translate(78,63) rotate(-34)"><path d="M-12-1.4c.6-3.2 5-4.8 10.6-4.6 6.4.2 12.4 2 13.8 4.8 1 2-1.4 5.2-6.4 6.8-5.6 1.8-12 1.4-15.6-1-1.8-1.2-2.6-3.6-2.4-6Z" fill="#22a699"></path></g>
    <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z" fill="#f4a63a"></path></g>
    <g transform="translate(20,60) rotate(62)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z" fill="#ef6f9b"></path></g>
    <g transform="translate(84,20) rotate(-8)"><path d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z" fill="#4cae4f"></path></g>
  </g>
  <g fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5" stroke-dasharray="4 3">
    <rect x="34" y="45" width="87" height="53" rx="2"></rect>
    <rect x="164" y="46" width="39" height="31" rx="2"></rect>
    <rect x="143" y="111" width="59" height="45" rx="2"></rect>
  </g>
  <rect x="56" y="120" width="95" height="65" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2.5"></rect>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--md-default-fg-color--light)">
    <text x="36" y="41">1</text>
    <text x="166" y="42">2</text>
    <text x="145" y="107">4</text>
  </g>
  <text x="58" y="116" style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--ngio-magenta)">3</text>
  <text x="16" y="216" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">each object carries an integer id</text>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M228 112h30M252 107l6 5-6 5"></path>
    <path d="M446 112h30M470 107l6 5-6 5"></path>
  </g>

  <rect x="274" y="28" width="160" height="140" rx="3" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <rect x="275" y="99" width="158" height="34" style="fill:var(--ngio-accent-soft)"></rect>
  <path d="M274 63h160M274 99h160M274 133h160" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color--light)">
    <text x="288" y="50">label</text>
    <text x="352" y="50">bounding box</text>
  </g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color)">
    <text x="288" y="86">1</text><text x="352" y="86">87 × 53</text>
    <text x="288" y="121">3</text><text x="352" y="121">95 × 65</text>
    <text x="288" y="155">4</text><text x="352" y="155">59 × 45</text>
  </g>
  <text x="274" y="216" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">one row per object, in world units</text>

  <rect x="492" y="28" width="132" height="104" rx="3" style="fill:var(--ngio-sunk)"></rect>
  <g transform="translate(558,79) scale(2.35)"><g transform="rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z" fill="#f4a63a"></path></g></g>
  <rect x="493" y="29" width="130" height="102" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2.5"></rect>
  <text x="492" y="156" style="font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:var(--md-default-fg-color)">label 3</text>
  <text x="492" y="216" style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">its own small array</text>
        </svg>
</div>

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:setup"
```

Like the `Image` and `Label` objects, a `MaskedImage` is initialised from an `OME-Zarr Container` object, using the `get_masked_image` method.

Create a masked image from the `nuclei` label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_masked_image"
```

Since `MaskedImage` is a subclass of `Image`, you can use every method available on `Image` objects.

The two most notable exceptions are `get_roi_as_numpy` (or `get_roi_as_dask`) and `set_roi`, which now take an integer `label` instead of a `roi` object.

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:masked_roi_numpy"
```

```python exec="true" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_helpers"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_masked_roi"
```

You can also use the `zoom_factor` argument to get more context around the ROI.
For example, zoom out the ROI by a factor of `2`:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:masked_roi_zoom"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_masked_roi_zoom"
```

## Masked operations

In addition to the `get_roi_as_numpy` method, the `MaskedImage` class also provides a masked operation method that allows you to perform reading and writing only on the masked pixels.

For these operations, use the `get_roi_masked` and `set_roi_masked` methods.
For example, use `get_roi_masked` to get the masked data for a specific label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_roi_masked"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_get_roi_masked"
```

Use the `set_roi_masked` method to set the masked data for a specific label:

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:set_roi_masked"
```

```python exec="true" html="1" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:plot_after_set_roi_masked"
```

## Masked labels

The `MaskedLabel` class is a subclass of [`Label`][ngio.Label] and provides the same functionality as the `MaskedImage` class.

Create a masked label from an `OME-Zarr Container` object using the `get_masked_label` method.

```python exec="true" source="material-block" session="masked_images"
--8<-- "docs/snippets/getting_started/masked_images.py:get_masked_label"
```

## Next steps

- [HCS plates](5_hcs.md) — scale up from a single image to a whole plate.
- [Iterators](6_iterators.md) — process every object or region without writing the loop yourself.
