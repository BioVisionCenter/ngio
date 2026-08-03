---
description: "Work with high-content screening plates: rows, columns, acquisitions, wells and images."
---

# 5. HCS plates

**Navigate a whole high-content screening plate.**

An HCS plate is a grid of wells, each holding one or more images, possibly from different
acquisitions. The [`OmeZarrPlate`][ngio.OmeZarrPlate] class gives you the rows, columns and
acquisitions of the plate, and the images inside each well.

<!-- Figure 04 — plate, well, image -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 240" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f4t f4d">
  <title id="f4t">Plate, well, field of view</title>
  <desc id="f4d">A plate is a grid of wells addressed by row letter and column number. A well holds one image per acquisition. Each image is an OME-Zarr container of its own.</desc>

  <g style="fill:var(--ngio-sunk);stroke:var(--ngio-line)">
    <rect x="16.5" y="8.5" width="195" height="25" rx="6"></rect>
    <rect x="270.5" y="8.5" width="131" height="25" rx="6"></rect>
    <rect x="460.5" y="8.5" width="163" height="25" rx="6"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:12.5px;fill:var(--md-default-fg-color)">
    <text x="114" y="26">plate.zarr</text>
    <text x="336" y="26">well B / 03</text>
    <text x="542" y="26">image B / 03 / 0</text>
  </g>
  <g text-anchor="middle" style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">
    <text x="114" y="52">rows × columns of wells</text>
    <text x="336" y="52">one image per acquisition</text>
    <text x="542" y="52">pixels, labels and ROIs</text>
  </g>

  <rect x="16" y="66" width="196" height="152" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <g style="font-family:'JetBrains Mono',monospace;font-size:9px;fill:var(--md-default-fg-color--lighter)">
    <text x="41" y="86" text-anchor="middle">1</text><text x="71" y="86" text-anchor="middle">2</text><text x="101" y="86" text-anchor="middle">3</text><text x="131" y="86" text-anchor="middle">4</text><text x="161" y="86" text-anchor="middle">5</text><text x="191" y="86" text-anchor="middle">6</text>
    <text x="28" y="104">A</text><text x="28" y="134">B</text><text x="28" y="164">C</text><text x="28" y="194">D</text>
  </g>
  <g style="fill:var(--md-default-fg-color--lightest)">
    <circle cx="41" cy="100" r="9"></circle><circle cx="71" cy="100" r="9"></circle><circle cx="101" cy="100" r="9"></circle><circle cx="131" cy="100" r="9"></circle><circle cx="161" cy="100" r="9"></circle><circle cx="191" cy="100" r="9"></circle>
    <circle cx="41" cy="130" r="9"></circle><circle cx="71" cy="130" r="9"></circle><circle cx="131" cy="130" r="9"></circle><circle cx="161" cy="130" r="9"></circle><circle cx="191" cy="130" r="9"></circle>
    <circle cx="41" cy="160" r="9"></circle><circle cx="71" cy="160" r="9"></circle><circle cx="101" cy="160" r="9"></circle><circle cx="131" cy="160" r="9"></circle><circle cx="161" cy="160" r="9"></circle><circle cx="191" cy="160" r="9"></circle>
    <circle cx="41" cy="190" r="9"></circle><circle cx="71" cy="190" r="9"></circle><circle cx="101" cy="190" r="9"></circle><circle cx="131" cy="190" r="9"></circle><circle cx="161" cy="190" r="9"></circle><circle cx="191" cy="190" r="9"></circle>
  </g>
  <circle cx="101" cy="130" r="9" style="fill:var(--ngio-accent)"></circle>
  <circle cx="101" cy="130" r="13" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5" opacity=".45"></circle>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M224 142h30M248 137l6 5-6 5"></path>
    <path d="M414 142h30M438 137l6 5-6 5"></path>
  </g>

  <rect x="270" y="66" width="132" height="152" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <g style="fill:var(--ngio-blue)">
    <rect x="284" y="82" width="52" height="52" rx="3"></rect>
    <rect x="346" y="82" width="52" height="52" rx="3" opacity=".45"></rect>
    <rect x="284" y="144" width="52" height="52" rx="3" opacity=".45"></rect>
    <rect x="346" y="144" width="52" height="52" rx="3" opacity=".45"></rect>
  </g>
  <rect x="281" y="79" width="58" height="58" rx="5" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5"></rect>

  <defs>
    <filter id="n4g" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch"></feTurbulence><feColorMatrix type="saturate" values="0"></feColorMatrix></filter>
    <g id="n4cells" fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.7" stroke-opacity=".92">
      <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path><g fill="#ffffff" stroke="none"><circle cx="-9" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-1" cy="3" r="1" opacity="0.7"></circle><circle cx="7" cy="-2.5" r="1.1" opacity="0.75"></circle><circle cx="13" cy="1.5" r="0.8" opacity="0.6"></circle></g></g>
      <g transform="translate(55,34) rotate(6)"><path d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-4" cy="-3" r="1.2" opacity="0.8"></circle><circle cx="3" cy="2" r="1" opacity="0.65"></circle><circle cx="6" cy="-3" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(78,63) rotate(-34)"><path d="M-12-1.4c.6-3.2 5-4.8 10.6-4.6 6.4.2 12.4 2 13.8 4.8 1 2-1.4 5.2-6.4 6.8-5.6 1.8-12 1.4-15.6-1-1.8-1.2-2.6-3.6-2.4-6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-1" r="1.1" opacity="0.75"></circle><circle cx="2" cy="1.5" r="0.9" opacity="0.6"></circle><circle cx="8" cy="-1" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-11" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-3" cy="3" r="1.1" opacity="0.7"></circle><circle cx="6" cy="-3" r="1.2" opacity="0.75"></circle><circle cx="14" cy="2" r="0.9" opacity="0.6"></circle><circle cx="1" cy="-6" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(20,60) rotate(62)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-2" r="1.2" opacity="0.8"></circle><circle cx="2" cy="2.5" r="1" opacity="0.65"></circle><circle cx="8" cy="-1.5" r="0.9" opacity="0.6"></circle></g></g>
      <g transform="translate(84,20) rotate(-8)"><path d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path><g fill="#ffffff" stroke="none"><circle cx="-2" cy="-2" r="1" opacity="0.7"></circle><circle cx="3" cy="2" r="0.9" opacity="0.6"></circle></g></g>
    </g>
    <g id="n4speck" fill="#ffffff">
      <circle cx="12" cy="14" r=".7" opacity=".4"></circle><circle cx="38" cy="10" r=".5" opacity=".28"></circle><circle cx="56" cy="34" r=".8" opacity=".32"></circle><circle cx="18" cy="44" r=".6" opacity=".22"></circle><circle cx="72" cy="42" r=".5" opacity=".3"></circle><circle cx="88" cy="24" r=".7" opacity=".24"></circle><circle cx="34" cy="66" r=".6" opacity=".28"></circle><circle cx="52" cy="72" r=".5" opacity=".2"></circle><circle cx="78" cy="70" r=".8" opacity=".3"></circle><circle cx="14" cy="58" r=".5" opacity=".18"></circle><circle cx="92" cy="86" r=".6" opacity=".26"></circle><circle cx="44" cy="92" r=".7" opacity=".22"></circle>
    </g>
    <clipPath id="n4p"><rect x="460" y="66" width="164" height="152" rx="3"></rect></clipPath>
  </defs>
  <g clip-path="url(#n4p)">
    <rect x="460" y="66" width="164" height="152" fill="#151d21"></rect>
    <use href="#n4cells" transform="translate(460,66) scale(1.64,1.52)"></use>
    <use href="#n4speck" transform="translate(460,66) scale(1.64,1.52)"></use>
    <rect x="460" y="66" width="164" height="152" filter="url(#n4g)" opacity=".10" style="mix-blend-mode:screen"></rect>
    <g transform="translate(460,66) scale(1.64,1.52)" opacity=".72">
      <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z" fill="#4cae4f"></path></g>
      <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z" fill="#f4a63a"></path></g>
    </g>
  </g>
  <path d="M515 66v152M570 66v152M460 117h164M460 168h164" stroke="#ffffff" stroke-width="1.1" opacity=".26"></path>
  <rect x="476" y="84" width="90" height="60" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5" stroke-dasharray="4 3"></rect>
        </svg>
</div>

Open an `OmeZarrPlate` object.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:setup"
```

This example plate is very small and contains only a single well.

## Plate overview

The `OmeZarrPlate` object gives you a high-level overview of the plate through three properties:

=== "Columns"
    Show the columns in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_columns"
    ```
=== "Rows"
    Show the rows in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_rows"
    ```
=== "Acquisitions"
    Show the acquisition ids:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_acquisitions"
    ```

## Retrieving the path to the images

The `OmeZarrPlate` object provides multiple methods to retrieve the path to the images in the plate.

=== "All image paths"
    This will return the paths to all images in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:images_paths"
    ```

=== "All well paths"
    This will return the paths to all wells in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:wells_paths"
    ```

=== "All image paths in a well"
    This will return the paths to all images in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:well_images_paths"
    ```

## Getting the images

`get_well_images` takes the row and column of a well and returns a dictionary mapping each image path to its [`OmeZarrContainer`][ngio.OmeZarrContainer].

=== "All images"
    Get all images in the plate:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_images"
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "All images in a well"
    Get all images in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_well_images"
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "Specific image"
    Get a specific image in a well:
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_image"
    ```
    This will return the `OmeZarrContainer` object for the image in the well.

=== "Filter by acquisition"
    In these methods, you can also filter the images by acquisition. When available, the `acquisition` parameter can be used to filter the images by acquisition id.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:get_well_images_by_acquisition"
    ```
    `acquisition` is optional: omit it and every image in the well is returned. Pass an acquisition id that the plate does not define — as on this example plate, which carries no acquisition metadata — and you get an empty dictionary back.

## Creating a plate

ngio provides a utility function to create a plate.

The first step is to create a list of `ImageInWellPath` objects. Each `ImageInWellPath` object contains the path to the image and the corresponding well.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:image_in_well_paths"
```

!!! note
    The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.

Then, you can create the plate using the `create_empty_plate` function.

```python exec="true" source="material-block" session="hcs_plate"
--8<-- "docs/snippets/getting_started/hcs.py:create_empty_plate"
```

This has created a new empty plate with the metadata correctly set. But no images have been added yet.

### Modifying the plate

You can add or remove images.

=== "Add images"
    To add images to the plate, use the `add_image` method. It takes the row and column of the well and the path to the image within it.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_add_image"
    ```
    This will add a new image to the plate and well metadata.
    !!! note
        The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.
    !!! warning
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_add_image` method instead.

        `atomic_add_image` serialises the update behind an OS file lock, so it holds across threads and processes on one machine. It requires a **local store** — on a remote store there is no lock to take and it raises `NgioValueError` — and on a shared network filesystem it is only as reliable as the mount's `flock` support. On **Windows** the lock is best-effort and warns, because `filelock` can hand the same lock to two workers at once: a single writer is safe, but concurrent ones can still lose an update, so run those on Linux or macOS.

=== "Remove images"
    To remove images from the plate, use the `remove_image` method. It takes the same arguments as `add_image`.
    ```python exec="true" source="material-block" session="hcs_plate"
    --8<-- "docs/snippets/getting_started/hcs.py:plate_remove_image"
    ```
    This will remove the image metadata from the plate and well metadata.
    !!! warning
        No data will be removed from the store. If an image is saved in the store it will remain there.
        Also the metadata will only be removed from the plate.well metadata. The number of columns and rows will not be updated.
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_remove_image` method instead, under the same store and platform limits as `atomic_add_image` above.

## Next steps

- [Iterators](6_iterators.md) — build pipelines that scale across a plate.
- [HCS exploration tutorial](../tutorials/hcs_exploration.md) — a worked example on real plate data.
- [HCS API reference](../api/hcs.md) — `OmeZarrPlate` and `OmeZarrWell`.
