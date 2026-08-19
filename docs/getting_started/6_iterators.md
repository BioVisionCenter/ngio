---
description: The five ngio iterators for building scalable image-processing pipelines.
---

# 6. Iterators

**Process an image region by region without writing the loop.**

When building image processing pipelines it is often useful to iterate over specific regions of the image, for example to process the image in smaller tiles or to process only specific regions of interest (ROIs). Iterators also let you set broadcasting rules for the iteration, for example to iterate over all z-planes or over all timepoints.

<!-- Figure 05 — how an iterator walks -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 232" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f5t f5d">
  <title id="f5t">How an iterator walks an image</title>
  <desc id="f5d">A ROI table names the regions. For each region the iterator reads that part of the input, applies your function, and writes the result into the output, one region at a time.</desc>
  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)">
    <rect x="24.5" y="4.5" width="15" height="14" rx="3"></rect>
    <rect x="184.5" y="4.5" width="15" height="14" rx="3"></rect>
    <rect x="344.5" y="4.5" width="15" height="14" rx="3"></rect>
    <rect x="496.5" y="4.5" width="15" height="14" rx="3"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)">
    <text x="32" y="15">1</text><text x="192" y="15">2</text><text x="352" y="15">3</text><text x="504" y="15">4</text>
  </g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="46" y="16">ROI TABLE</text>
    <text x="206" y="16">INPUT</text>
    <text x="366" y="16">YOUR FUNCTION</text>
    <text x="518" y="16">OUTPUT</text>
  </g>

  <rect x="24" y="28" width="110" height="132" rx="3" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <rect x="25" y="29" width="108" height="32" style="fill:var(--ngio-accent-soft)"></rect>
  <path d="M24 61h110M24 94h110M24 127h110" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>
  <g style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color)">
    <text x="38" y="49">region 1</text>
    <text x="38" y="82">region 2</text>
    <text x="38" y="115">region 3</text>
    <text x="38" y="148">…</text>
  </g>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M144 94h30M168 89l6 5-6 5"></path>
    <path d="M314 94h30M338 89l6 5-6 5"></path>
    <path d="M456 94h30M480 89l6 5-6 5"></path>
  </g>

  <defs>
    <clipPath id="n5p"><rect x="184" y="28" width="120" height="132" rx="3"></rect></clipPath>
    <clipPath id="n5o"><path d="M496 28h60v44h-60z"></path></clipPath>
    <filter id="n5g" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch"></feTurbulence><feColorMatrix type="saturate" values="0"></feColorMatrix></filter>
  </defs>
  <g clip-path="url(#n5p)">
    <rect x="184" y="28" width="120" height="132" fill="#151d21"></rect>
    <g transform="translate(184,28) scale(1.2,1.32)" fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.4" stroke-opacity=".92">
      <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path><g fill="#ffffff" stroke="none"><circle cx="-9" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-1" cy="3" r="1" opacity="0.7"></circle><circle cx="7" cy="-2.5" r="1.1" opacity="0.75"></circle><circle cx="13" cy="1.5" r="0.8" opacity="0.6"></circle></g></g>
      <g transform="translate(55,34) rotate(6)"><path d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-4" cy="-3" r="1.2" opacity="0.8"></circle><circle cx="3" cy="2" r="1" opacity="0.65"></circle><circle cx="6" cy="-3" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(78,63) rotate(-34)"><path d="M-12-1.4c.6-3.2 5-4.8 10.6-4.6 6.4.2 12.4 2 13.8 4.8 1 2-1.4 5.2-6.4 6.8-5.6 1.8-12 1.4-15.6-1-1.8-1.2-2.6-3.6-2.4-6Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-1" r="1.1" opacity="0.75"></circle><circle cx="2" cy="1.5" r="0.9" opacity="0.6"></circle><circle cx="8" cy="-1" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-11" cy="-2" r="1.3" opacity="0.85"></circle><circle cx="-3" cy="3" r="1.1" opacity="0.7"></circle><circle cx="6" cy="-3" r="1.2" opacity="0.75"></circle><circle cx="14" cy="2" r="0.9" opacity="0.6"></circle><circle cx="1" cy="-6" r="0.8" opacity="0.55"></circle></g></g>
      <g transform="translate(20,60) rotate(62)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path><g fill="#ffffff" stroke="none"><circle cx="-6" cy="-2" r="1.2" opacity="0.8"></circle><circle cx="2" cy="2.5" r="1" opacity="0.65"></circle><circle cx="8" cy="-1.5" r="0.9" opacity="0.6"></circle></g></g>
      <g transform="translate(84,20) rotate(-8)"><path d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path><g fill="#ffffff" stroke="none"><circle cx="-2" cy="-2" r="1" opacity="0.7"></circle><circle cx="3" cy="2" r="0.9" opacity="0.6"></circle></g></g>
    </g>
    <g fill="#ffffff"><circle cx="200" cy="46" r=".7" opacity=".4"></circle><circle cx="264" cy="40" r=".6" opacity=".28"></circle><circle cx="230" cy="100" r=".7" opacity=".3"></circle><circle cx="288" cy="86" r=".6" opacity=".26"></circle><circle cx="210" cy="146" r=".6" opacity=".24"></circle><circle cx="278" cy="140" r=".7" opacity=".26"></circle></g>
    <rect x="184" y="28" width="120" height="132" filter="url(#n5g)" opacity=".10" style="mix-blend-mode:screen"></rect>
  </g>
  <path d="M244 28v132M184 72h120M184 116h120" stroke="#ffffff" stroke-width="1.2" opacity=".3"></path>
  <rect x="185" y="29" width="58" height="42" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2.5"></rect>

  <rect x="352" y="74" width="96" height="40" rx="10" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5"></rect>
  <text x="400" y="99" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color)">process()</text>

  <rect x="496.5" y="28.5" width="59" height="43" rx="2" style="fill:var(--ngio-sunk);stroke:var(--ngio-line-strong)"></rect>
  <g clip-path="url(#n5o)">
    <g transform="translate(496,28) scale(1.2,1.32)">
      <g transform="translate(30,26) rotate(-14)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z" fill="#4cae4f"></path></g>
      <g transform="translate(55,34) rotate(6)"><path d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z" fill="#7c6bd6"></path></g>
      <g transform="translate(78,63) rotate(-34)"><path d="M-12-1.4c.6-3.2 5-4.8 10.6-4.6 6.4.2 12.4 2 13.8 4.8 1 2-1.4 5.2-6.4 6.8-5.6 1.8-12 1.4-15.6-1-1.8-1.2-2.6-3.6-2.4-6Z" fill="#22a699"></path></g>
      <g transform="translate(44,74) rotate(22)"><path d="M-20-1.4c-.6-6.6 6.4-10.2 15.4-9.6 10.6.6 20 3.4 23.6 7.6 2.2 2.6 1.2 6.8-2.6 10-5.4 4.4-15.4 5.8-23.8 4.6C-14.6 10.2-19.4 6.6-20-1.4Z" fill="#f4a63a"></path></g>
      <g transform="translate(20,60) rotate(62)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z" fill="#ef6f9b"></path></g>
      <g transform="translate(84,20) rotate(-8)"><path d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z" fill="#4cae4f"></path></g>
    </g>
  </g>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5" stroke-dasharray="4 3">
    <rect x="556.75" y="28.75" width="58.5" height="42.5" rx="2"></rect>
    <rect x="496.75" y="72.75" width="58.5" height="42.5" rx="2"></rect>
    <rect x="556.75" y="72.75" width="58.5" height="42.5" rx="2"></rect>
    <rect x="496.75" y="116.75" width="58.5" height="42.5" rx="2"></rect>
    <rect x="556.75" y="116.75" width="58.5" height="42.5" rx="2"></rect>
  </g>

  <path d="M586 168v28H214v-26M209 176l5-6 5 6" fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5" stroke-dasharray="5 4"></path>
  <text x="400" y="216" text-anchor="middle" style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color--light)">repeat for every region the table names</text>
        </svg>
</div>

ngio provides five basic `Iterator` classes, all imported from `ngio.iterators` (or from
the top-level `ngio` namespace):

<!-- Figure 06 — which iterator do I want -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 298" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f6t f6d">
  <title id="f6t">The five iterators, by what they take and return</title>
  <desc id="f6d">Segmentation takes an image and returns a label. Masked segmentation takes an image and a label and returns a label. Image processing takes an image and returns an image. Feature extraction takes an image and a label and returns a table. Object detection takes an image and returns a table of boxes.</desc>

  <g style="stroke:var(--ngio-line)"><path d="M16 52h608M16 100h608M16 148h608M16 196h608"></path></g>

  <g style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">
    <text x="16" y="31">SegmentationIterator</text>
    <text x="16" y="79">MaskedSegmentationIterator</text>
    <text x="16" y="127">ImageProcessingIterator</text>
    <text x="16" y="175">FeatureExtractorIterator</text>
    <text x="16" y="223">ObjectDetectionIterator</text>
  </g>

  <defs>
    <g id="n6i">
      <rect width="22" height="22" rx="3" fill="#151d21"></rect>
      <g fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.8" stroke-opacity=".85">
        <g transform="translate(9,8) rotate(-18) scale(.42)"><path d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path></g>
        <g transform="translate(15,16) rotate(12) scale(.4)"><path d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path></g>
      </g>
      <circle cx="4" cy="17" r=".7" fill="#ffffff" opacity=".5"></circle>
      <circle cx="18.5" cy="5" r=".6" fill="#ffffff" opacity=".4"></circle>
    </g>
    <g id="n6l">
      <rect x=".5" y=".5" width="21" height="21" rx="3" style="fill:var(--ngio-sunk);stroke:var(--ngio-line-strong)"></rect>
      <ellipse cx="8.5" cy="8" rx="6" ry="4.5" transform="rotate(-18 8.5 8)" fill="#4cae4f"></ellipse>
      <ellipse cx="15" cy="16" rx="5" ry="3.8" transform="rotate(10 15 16)" fill="#7c6bd6"></ellipse>
      <ellipse cx="5" cy="17" rx="3.2" ry="2.4" fill="#f4a63a"></ellipse>
    </g>
  </defs>
  <use href="#n6i" x="238" y="15"></use>
  <use href="#n6i" x="238" y="63"></use>
  <use href="#n6i" x="238" y="111"></use>
  <use href="#n6i" x="238" y="159"></use>
  <use href="#n6i" x="238" y="207"></use>
  <use href="#n6l" x="266" y="63"></use>
  <use href="#n6l" x="266" y="159"></use>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M300 26h26M320 21l6 5-6 5"></path>
    <path d="M300 74h26M320 69l6 5-6 5"></path>
    <path d="M300 122h26M320 117l6 5-6 5"></path>
    <path d="M300 170h26M320 165l6 5-6 5"></path>
    <path d="M300 218h26M320 213l6 5-6 5"></path>
  </g>

  <use href="#n6l" x="338" y="15"></use>
  <use href="#n6l" x="338" y="63"></use>
  <use href="#n6i" x="338" y="111"></use>
  <rect x="338.75" y="159.75" width="20.5" height="20.5" rx="2.5" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <path d="M338 167h22M345 160v20M352 160v20" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>
  <rect x="338.75" y="207.75" width="20.5" height="20.5" rx="2.5" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <path d="M338 215h22M345 208v20M352 208v20" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>

  <g style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color--light)">
    <text x="384" y="31">an image in, a new label out</text>
    <text x="384" y="79">the same, restricted to one mask</text>
    <text x="384" y="127">an image in, a new image out</text>
    <text x="384" y="175">read only — measurements out</text>
    <text x="384" y="223">read only — detected boxes out</text>
  </g>

  <path d="M16 254h608" style="stroke:var(--ngio-line)"></path>
  <use href="#n6i" transform="translate(16,264) scale(0.64)"></use>
  <use href="#n6l" transform="translate(104,264) scale(0.64)"></use>
  <rect x="192.75" y="264.75" width="12.5" height="12.5" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></rect>
  <path d="M192 269h14M197 264v14M201 264v14" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></path>
  <g style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">
    <text x="38" y="275">image</text>
    <text x="126" y="275">labels</text>
    <text x="214" y="275">table</text>
  </g>
        </svg>
</div>


* The `SegmentationIterator` is designed to build segmentation pipelines, where an input image is processed to produce a segmentation mask. For a worked example, see the [image segmentation tutorial](../tutorials/image_segmentation.md).
* The `MaskedSegmentationIterator` is similar to the `SegmentationIterator`, but it uses a masking ROI table to restrict the segmentation to masks. This is useful when you want to segment only specific regions of the image, for example, segmenting cells only within a specific tissue region. For a worked example, see the [image segmentation tutorial](../tutorials/image_segmentation.md).
* The `ImageProcessingIterator` is designed to build image processing pipelines, where an input image is processed to produce a new image. For a worked example, see the [image processing tutorial](../tutorials/image_processing.md).
* The `FeatureExtractorIterator` is a read-only iterator designed to iterate over pairs of images and labels to extract features from the image based on the labels. Its `reduce_to_table` runs a measurement over every region and returns the joined results as a single feature table — parallel per region via `mapper=`, stored by your own `add_table` call. For a worked example, see the [feature extraction tutorial](../tutorials/feature_extraction.md).
* The `ObjectDetectionIterator` runs a detector (a YOLO model, a maxima finder) tile by tile and returns the found objects as a single ROI table — see [Detecting objects into a ROI table](#detecting-objects-into-a-roi-table).

## Building one

Every iterator is constructed from the images it reads and writes, then narrowed. A fresh
iterator covers the whole image as a single region:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:setup"
```

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:build"
```

`product` replaces that single region with the ones a ROI table names — here the
microscope fields of view:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:product"
```

The regions are ordinary [`Roi`][ngio.Roi] objects, so you can inspect them before
processing anything:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:inspect"
```

Beyond a ROI table, four tiling calls reshape the regions, and each name says what it
tiles by. `by_grid(size_x=..., size_y=..., ...)` lays a regular grid of the sizes you
ask for; where the grid does not divide the axis, its `tail=` policy decides what
happens to the leftover — `"clip"` (the default) shrinks the last tile to the border,
`"balance"` re-splits the last two tiles so a thin overhang never yields a thin tile
(100 px at 32 gives `32, 32, 18, 18` rather than `32, 32, 32, 4`), `"shift"` slides
the last tile back to stay full-size (it then overlaps its neighbour — fine for
detection or a merge; a plain parallel write schedules the overlap into a separate
wave, see below), and `"drop"` discards it.
`by_blocks(num_x=..., num_y=...)` is the complement — you say how many tiles, not how
big, and the partition is balanced by construction. `by_chunks()` tiles by the *input*
image's chunk grid, the natural unit of reading; `by_write_units()` tiles by the
*output*'s write granularity — the shard shape when the output is sharded, the chunk
shape otherwise — which makes parallel writes collision-free by construction, so a
parallel `map` runs as a single fully-parallel wave.

From here you would call `map` or iterate with `iter_as_numpy` to do the work;
the [image processing tutorial](../tutorials/image_processing.md) carries this through to
a written result.

More complete examples can be found in the [Fractal tasks template](https://github.com/fractal-analytics-platform/fractal-tasks-template).

## Parallel mapping

`map` runs one ROI at a time by default, and that stays the default — parallel writing is explicit opt-in. Concurrency belongs to the *mapper*: pass one, and it sizes its own pool.

The examples from here on run on a small synthetic image — two bright blobs, one of them deliberately crossing a tile boundary:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:synthetic_setup"
```

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:mapper_demo"
```

`ThreadedMapper` is the fit for IO-bound work and for funcs that release the GIL (most numpy/scipy do); `"auto"` sizes the pool for round-trip-bound work. For pure-Python, GIL-holding funcs use `ProcessMapper(max_workers=...)` instead — the func must be picklable (a module-level function, not a lambda), and the store must not be in-memory.

Before fanning out, both parallel mappers plan every ROI's *write footprint* — the chunks (or shards, when the output is sharded) it will write on the **output** image — into conflict-free **waves**: ROIs sharing a write unit land in different waves, and the waves run back to back, each at full pool width. Within a wave the footprints are disjoint, which is what makes the parallel writes safe without any locking, for threads and processes alike: each chunk or shard object has exactly one writer at a time. A tiling with no shared write units — `by_write_units()`, as in the demo above — runs as a single fully-parallel wave; more sharing means more waves, down to an effectively serial schedule when every ROI collides (ngio logs a warning there rather than refusing). This is what lets a masked iterator parallelize out of the box: per-object bounding boxes routinely share chunks even when the boxes themselves do not overlap, and the wave count stays around the worst chunk's object multiplicity.

One caveat for ROIs whose *pixels* genuinely overlap (a `by_grid` with a stride below the size, or a `"shift"` tail, on a writing iterator): such writes always land in different waves, but the wave order may differ from a serial run's ROI order, so which write wins the shared pixels can differ from serial. Mask-protected and pixel-disjoint writes are order-independent, and an order-independent `merge=` (`"max"`, `"min"`, `"sum"`) never notices.

Two contracts are yours: under threads the `func` must be thread-safe, and under processes it must be picklable. ngio's side — the per-ROI readers and writers — is safe in both settings. The dask iterator surface (`iter_as_dask`, `map_as_dask`, `reduce_as_dask`) is deprecated and will be removed in ngio=1.2; for lazy whole-region access use `Image.get_as_dask` instead.

For per-ROI measurement without writing anything, use `reduce` — it returns one result per ROI, in ROI order, and takes the same `mapper` argument:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:reduce_demo"
```

## Batched inference

For neural-network inference the per-call overhead usually dominates, and the model wants one batched `(B, ...)` input rather than one patch at a time. `BatchedMapper` does exactly that: it stacks up to `batch_size` patches into a single array, calls the func **once per batch**, and writes each result back to its own region:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:batched_demo"
```

This is the one mapper that changes the func's contract: it receives a stacked batch with a leading batch axis and must return an array with the same leading axis. Everything else composes as usual — tilings are ragged in general (border tiles clip, halos shrink at image edges, masked regions are arbitrary), so each batch is padded up to its per-axis maximum before stacking (`pad_mode`/`pad_values` choose how) and every output is sliced back to its region's true shape before the write; a halo is trimmed after that, exactly as under any other mapper; and `for_job` partitions batch their own share. Within a batch the reads fan out on a thread pool (`read_workers`), while the writes run serially on the calling thread — like the serial mapper, batched mapping is write-safe on any tiling, no wave planning involved.

## Distributed runs: selecting a partition

A mapper parallelizes within one machine. On a cluster — SLURM array tasks over a shared filesystem, where processes cannot coordinate — restrict each task to one partition of the work instead:

```python
n_jobs = 4
job_index = 0  # e.g. int($SLURM_ARRAY_TASK_ID)

iterator = SegmentationIterator(image, label, ...).by_chunks()
iterator.for_job(job_index, n_jobs=n_jobs).map(func)
```

and, in a dependent job once every array task has finished, the gather step:

```python
iterator = SegmentationIterator(image, label, ...).by_chunks()  # same construction
iterator.finalize()
```

`for_job` is a builder call like `by_grid` or `with_halo` — it returns a new iterator restricted to that partition's regions, and everything else reads as usual, `map(func, mapper=...)` included. It comes *last* in the chain (reshaping a restricted iterator refuses), and its `map` deliberately does **not** finalize: the pyramid resolve is the one global step, and it belongs to the single gather job — which, thanks to region-scoped consolidation, rebuilds only what the jobs wrote. Until the gather runs, only the iterated level is up to date.

Each job builds the identical iterator — construction is metadata-only and deterministic, so this is cheap — and derives the same partition on its own; there is nothing to hand from one job to another. The guarantee behind it is *embarrassing independence*: the units are grouped by their write-conflict components (the same adjacency the wave scheduler uses), and no write unit is ever shared between two partitions — so the jobs need no locks, no barriers, and no channel to one another, in any order and any overlap in time. Regions whose footprints conflict simply travel in the same partition, where the ordinary wave planning handles them.

Effective parallelism therefore equals the number of independent groups, which follows the **output's** chunking. Inspect it before submitting — `[it.for_job(i, n_jobs=n).partition_indices for i in range(n)]` — one fat list plus empties means the output chunking (or a tiling like `by_zyx`, which splits along t only), not the cluster, is the constraint; a single-chunk output is one group by construction, since a chunk is one atomic write object. Surplus partitions are harmless no-ops, and [`write_conflict_components`][ngio.iterators.write_conflict_components] makes the grouping auditable.

The requirements mirror the model: every job must use the same `n_jobs` and the same iterator construction; the store must not be in-memory (each process would write its own private copy). Read-only iterators refuse to partition — their gathers (feature coalescing, detection NMS) are global joins.

### The three-phase recipe: `prepare_jobs`

Schedulers like Fractal run distributed work as **init → parallel tasks → consolidate**, where the init task builds a *parallelization list* (one JSON of arguments per parallel task). `prepare_jobs` is that init step: it performs any setup the iterator needs — always wiping stale scratch state from earlier runs first — and returns the list, with empty partitions already dropped:

```python
# init task
iterator = SegmentationIterator(image, label, ...).by_chunks()
args_list = iterator.prepare_jobs(n_jobs=4)
# -> [{"job_index": 0, "n_jobs": 4}, {"job_index": 1, "n_jobs": 4}, ...]

# parallel task, once per entry
iterator = SegmentationIterator(image, label, ...).by_chunks()
iterator.for_job(**args).map(func)

# consolidate task, after all parallel tasks
iterator = SegmentationIterator(image, label, ...).by_chunks()
iterator.finalize()
```

For a plain writing iterator `prepare_jobs` is optional — the two-step recipe above works on its own. A **stitching** segmentation requires it: the scratch band arrays must exist, race-free, before any job banks into them, and the init step is the one safe moment to create them.

### Distributed stitching

With `prepare_jobs` in the recipe, `stitch=True` distributes too. Each job's `map` banks its tiles' seam bands into the shared scratch exactly as a local run would; the consolidate task's `finalize()` runs the one global resolve — seam scan, id union, renumbering to a dense `1..N` — then rebuilds the pyramid and removes the scratch. Three properties are worth knowing:

- **Band writes join the conflict graph.** Tiles whose bands would land in the same scratch chunk travel in the same job (and never share a wave locally), so band banking needs no locks anywhere. A tile grid aligned with the label's chunk grid still splits fully.
- **A failed job never destroys the others' bands.** Re-run just that job — banding is idempotent, the id offsets are derived from the global tile index — and gather as planned. A fresh `prepare_jobs` always starts from a clean slate.
- **Every step validates a plan fingerprint** stamped at init: change the tiling, halo, stitch config, or `n_jobs` between phases and the run fails loudly instead of resolving against the wrong bands.

The consolidate task is the one global step — the seam scan and relabel run single-node over the whole label — so distribution accelerates the segmentation itself, not the final reconciliation.

### Distributed measurement and detection

The read-only iterators end in a *global join* — one feature coalesce, one NMS pass — that per-job runs cannot reproduce piecewise (greedy NMS is not hierarchical: suppressing per job and then merging can keep different boxes than one global pass). Their distributed form therefore stores each job's **raw pre-join records** as a *partial*, and the consolidate step runs the single global join:

```python
# init task
iterator = FeatureExtractorIterator(image, label).by_grid(size_y=512, size_x=512)
args_list = iterator.prepare_jobs(n_jobs=4)

# parallel task, once per entry
iterator.for_job(**args).reduce_to_partial(measure)      # features
# iterator.for_job(**args).detect_to_partial(detector)   # detection

# consolidate task, after all parallel tasks
table = iterator.merge_partials()
container.add_table("measurements", table)               # storing stays yours
```

The result is bit-identical to a serial `reduce_to_table` / `detect` — including a **custom `coalesce`**, which runs once at merge time over the reconstructed per-ROI results (dicts normalized to DataFrames, a `label` index to a `label` column). Partials live in a transient `_ngio_partials` group beside the resolution levels, written through ngio's own table backends (so every store type and the retry policy apply), invisible to `list_tables`, and removed by the merge; the final table is registered only by your own `add_table` call. The merge refuses a half-finished run — a missing job errors instead of producing a plausible-looking, silently incomplete table — and the finished-table verbs (`reduce_to_table`, `detect`) refuse on a `for_job` slice, pointing at their partial counterparts.

## Halos: context without seams

Tiling an image and processing each tile independently leaves artifacts at the joins — a smoothing kernel at a tile edge has no neighbours to work with, and a segmentation cuts objects at the boundary. `with_halo` fixes that by reading a margin around each ROI and writing only the ROI back:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:halo_demo"
```

`smooth` receives the grown region and must return it grown too; the border is cropped off before the write, so it never lands on disk. Margins are in pixels and clip at the image borders, so an edge tile simply grows on the sides where there is room.

The ROIs themselves do not move, which is the whole point of doing this on the read side: write footprints are unchanged, so a haloed iterator parallelizes exactly as far as it did without one. Overlapping *writes* would have to be serialized; overlapping reads cost nothing.

Read the same trick backwards and it is a "trim": if you want each tile's outer margin discarded rather than written, that is exactly a halo of that width.

## Merging instead of overwriting

By default a write replaces what is on disk. `merge=` combines with it instead:

```python
image.set_roi(roi, patch, merge="max")
```

`"max"`, `"min"` and `"sum"` are commutative and associative, so overlapping regions give the same answer whatever order they are written in. `"keep_nonzero"` ("the last nonzero write wins") and a custom `(existing, patch, ctx) -> array` rule do depend on the order.

The merge is a separate argument rather than an entry in `transforms=`, and the distinction matters. A transform is a function of the patch alone, which is why the chain composes and inverts with no rules about order or position. A merge also depends on what is already there, so it runs once, after the chain — by which point the patch is in the array's own space and the destination is read raw. Both sides are in the same space, which is what makes the comparison meaningful and keeps untouched pixels byte-identical instead of round-tripping them through a transform and back.

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

The offset is derived from the block index rather than counted up as regions are processed. That is what makes it work in parallel: there is no shared counter to synchronize, it survives `ProcessMapper`, and it is idempotent — re-running a region after a failure reproduces exactly the ids it wrote before, where a counter would hand out a fresh set and strand the old one.

Being an ordinary transform, it composes with a merge:

```python
label.set_roi(
    roi, patch, transforms=[UniqueLabelsTransform(1000, block_index=4)], merge="max"
)
```

## Stitching a tiled segmentation

Segmenting tile by tile leaves an object that crosses a boundary as two objects with two ids — and, because every tile numbers its objects from 1, leaves ids that mean nothing outside their own tile. `stitch=True` fixes both:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:stitch_demo"
```

The halo is required, and the reason is the criterion. Stitching joins two ids when the two tiles' predictions **overlap**, not when their objects merely touch across the cut. Two distinct objects that abut at a tile boundary are adjacent but do not overlap, so an adjacency rule would merge them and an overlap rule does not. Each tile's halo is what gives it an opinion about the strip its neighbour owns, and comparing the two opinions is what the stitch does.

Those opinions have to be kept somewhere, since a tile writes only its core — so the map also banks each tile's halo band into a small transient array, removed once the stitch resolves. Write footprints on the label itself are unchanged, so a stitching iterator parallelizes exactly as far as it did without one.

Tune it with `StitchConfig`:

```python
from ngio.iterators import StitchConfig

iterator = SegmentationIterator(
    image, label, stitch=StitchConfig(iou_threshold=0.5, block_size=50_000)
)
```

By default the halo bands are kept in a transient group inside the output label, which works under every mapper. `scratch_store` puts them elsewhere — often worth doing, since labels compress well and the bands are small:

```python
from zarr.storage import MemoryStore

StitchConfig(scratch_store=MemoryStore())
```

That keeps the output store untouched and leaves nothing behind if a run dies. The one restriction is `ProcessMapper`: a `MemoryStore` pickles by value, so each worker would bank its bands into a private copy — ngio refuses that rather than losing them silently.

`iou_threshold` is how much two tiles must agree before their ids are joined. The default errs towards leaving an object split rather than merging two that are not — an over-split label can be fixed downstream, a wrong merge cannot. `block_size` is how many ids each tile is given, and must exceed the largest count a single tile can produce.

Compaction is not exclusive to stitching — `label.relabel_sequential()` renumbers any label to a dense `1..N` on its own:

```python
label.relabel_sequential()
```

Either way the numbering is assigned in first-encounter order over the chunk grid rather than by sorting the existing ids. That keeps it to a single pass over the label, and means which object ends up as `1` follows the array rather than the tile it came from.

If a run is interrupted between the map and the resolve, the label holds a valid but over-split segmentation; re-running the resolve is safe, because it is idempotent.

## Detecting objects into a ROI table

Not every model produces a mask. An object detector — a YOLO network, a spot finder — reports **bounding boxes**, and the natural home for those is a ROI table, not a label image. The `ObjectDetectionIterator` runs a detector tile by tile and returns one `RoiTable` of the objects it found:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:detect_demo"
```

NMS is configured with `nms=NmsConfig(iou_threshold=..., score_column=...)` on the constructor, exactly as stitching is with `stitch=StitchConfig(...)`; a parallel `mapper=` on `detect` fans the tiles out like any `reduce`.

The detector sees one tile at a time and answers in the tile's own pixels: `(patch) -> list[Roi]`, each box built the way any ROI is — `Roi.from_values(slices={"x": (x0, width), "y": (y0, height)}, name=None, space="pixel", confidence=0.9)`. Boxes pin `x` and `y` (and optionally `z` for 3D boxes); `space="pixel"` is required, and a world-space box is refused — patch-local numbers in a world-labelled Roi would land every box in the wrong place silently. Whatever else the box carries as extra fields — confidence, class — rides along into the table unchanged, but `name` and `label` are refused: the iterator itself assigns them when it renumbers the survivors (put a class label in an extra field, e.g. `class_id`). The iterator does the bookkeeping the detector should not: it anchors each tile's boxes into the reference image's world coordinates, and it resolves the boundary problem.

The boundary problem is the sliding-window one. An object cut by a tile edge is seen only partially by either tile, so each tile reads a halo past its edge — this is the one read-only iterator on which `with_halo` is allowed, because there is no write to crop the margin from — and the object is seen whole by at least one of them. The cost is that both neighbours now report it, and the cure is standard **non-maximum suppression**, configured by `nms=NmsConfig(...)` exactly as stitching is by `stitch=StitchConfig(...)`: boxes overlapping at or above `iou_threshold` (default `0.5`) are one object, and the one ranked higher by the `score_column` (`"confidence"` by default; box volume when the detector reports no score) survives. Per-tile NMS inside the detector composes cleanly with this cross-tile pass. The survivors are renumbered to a dense `1..N` and returned; like `reduce_to_table`, nothing is written — storing the table is your `add_table` call.

Two contracts worth knowing. Every tile must report the same box dimensionality (all 2D or all 3D, scored or unscored) — mixtures raise. And a 2D detector on a 3D or timelapse image never has its boxes merged across the un-pinned axes: detections from different z-slabs or time points keep their tile's extent along those axes and are deduplicated only within it.

### Anchoring a local box yourself

The coordinate bookkeeping is one call you can also use in a custom flow or the manual pattern: `Roi.anchor` turns a region's ROI plus a patch-local pixel box into the absolute world ROI —

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:anchor_demo"
```

The `space` fields are what keep this honest: `anchor` refuses a world-space box (already absolute — anchoring it would double the offset) and a pixel-space region, so the classic silent frame mix-up raises instead. Axes the box does not pin inherit the region's extent.

## Next steps

- [Object detection tutorial](../tutorials/object_detection.md) — a spot finder through `detect`, end to end.
- [Image processing tutorial](../tutorials/image_processing.md) — an iterator applied end to end.
- [Image segmentation tutorial](../tutorials/image_segmentation.md) — segmentation and masked segmentation.
- [Feature extraction tutorial](../tutorials/feature_extraction.md) — `reduce_to_table` on a segmented image.
- [Iterators API reference](../api/iterators.md) — the full iterator API.
