---
description: The four ngio iterators for building scalable image-processing pipelines.
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

ngio provides four basic `Iterator` classes, all imported from `ngio.iterators` (or from
the top-level `ngio` namespace):

<!-- Figure 06 — which iterator do I want -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 250" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f6t f6d">
  <title id="f6t">The four iterators, by what they take and return</title>
  <desc id="f6d">Segmentation takes an image and returns a label. Masked segmentation takes an image and a label and returns a label. Image processing takes an image and returns an image. Feature extraction takes an image and a label and returns a table.</desc>

  <g style="stroke:var(--ngio-line)"><path d="M16 52h608M16 100h608M16 148h608"></path></g>

  <g style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">
    <text x="16" y="31">SegmentationIterator</text>
    <text x="16" y="79">MaskedSegmentationIterator</text>
    <text x="16" y="127">ImageProcessingIterator</text>
    <text x="16" y="175">FeatureExtractorIterator</text>
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
  <use href="#n6l" x="266" y="63"></use>
  <use href="#n6l" x="266" y="159"></use>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M300 26h26M320 21l6 5-6 5"></path>
    <path d="M300 74h26M320 69l6 5-6 5"></path>
    <path d="M300 122h26M320 117l6 5-6 5"></path>
    <path d="M300 170h26M320 165l6 5-6 5"></path>
  </g>

  <use href="#n6l" x="338" y="15"></use>
  <use href="#n6l" x="338" y="63"></use>
  <use href="#n6i" x="338" y="111"></use>
  <rect x="338.75" y="159.75" width="20.5" height="20.5" rx="2.5" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <path d="M338 167h22M345 160v20M352 160v20" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>

  <g style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color--light)">
    <text x="384" y="31">an image in, a new label out</text>
    <text x="384" y="79">the same, restricted to one mask</text>
    <text x="384" y="127">an image in, a new image out</text>
    <text x="384" y="175">read only — measurements out</text>
  </g>

  <path d="M16 206h608" style="stroke:var(--ngio-line)"></path>
  <use href="#n6i" transform="translate(16,216) scale(0.64)"></use>
  <use href="#n6l" transform="translate(104,216) scale(0.64)"></use>
  <rect x="192.75" y="216.75" width="12.5" height="12.5" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></rect>
  <path d="M192 221h14M197 216v14M201 216v14" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></path>
  <g style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">
    <text x="38" y="227">image</text>
    <text x="126" y="227">labels</text>
    <text x="214" y="227">table</text>
  </g>
        </svg>
</div>


* The `SegmentationIterator` is designed to build segmentation pipelines, where an input image is processed to produce a segmentation mask. For a worked example, see the [image segmentation tutorial](../tutorials/image_segmentation.md).
* The `MaskedSegmentationIterator` is similar to the `SegmentationIterator`, but it uses a masking ROI table to restrict the segmentation to masks. This is useful when you want to segment only specific regions of the image, for example, segmenting cells only within a specific tissue region. For a worked example, see the [image segmentation tutorial](../tutorials/image_segmentation.md).
* The `ImageProcessingIterator` is designed to build image processing pipelines, where an input image is processed to produce a new image. For a worked example, see the [image processing tutorial](../tutorials/image_processing.md).
* The `FeatureExtractorIterator` is a read-only iterator designed to iterate over pairs of images and labels to extract features from the image based on the labels. For a worked example, see the [feature extraction tutorial](../tutorials/feature_extraction.md).

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

From here you would call `map` or iterate with `iter_as_numpy` to do the work;
the [image processing tutorial](../tutorials/image_processing.md) carries this through to
a written result.

More complete examples can be found in the [Fractal tasks template](https://github.com/fractal-analytics-platform/fractal-tasks-template).

## Parallel mapping

`map` runs one ROI at a time by default, and that stays the default — parallel writing is explicit opt-in. Concurrency belongs to the *mapper*: pass one, and it sizes its own pool.

```python
from ngio import ProcessMapper, ThreadedMapper

# Threads: the fit for IO-bound work and for funcs that release the GIL
# (most numpy/scipy do). "auto" sizes the pool for round-trip-bound work.
iterator.map(run_segmentation, mapper=ThreadedMapper("auto"))

# Processes: the fit for pure-Python, GIL-holding funcs. The func must be
# picklable (a module-level function, not a lambda), and the store must not
# be in-memory.
iterator.map(run_segmentation, mapper=ProcessMapper(max_workers=8))
```

Before fanning out, both parallel mappers check every ROI's *write footprint* — the chunks (or shards, when the output is sharded) it will write on the **output** image. Disjoint footprints are what make the parallel writes safe without any locking, for threads and processes alike: each chunk or shard object has exactly one writer. If two ROIs share a write unit the mapper refuses with an error naming them; the fix it suggests, `by_chunks(grid="write")`, re-tiles the iterator on the output's write grid so collisions are impossible by construction:

```python
iterator = iterator.by_chunks(grid="write")
iterator.map(run_segmentation, mapper=ThreadedMapper("auto"))   # cannot collide
```

Two contracts are yours: under threads the `func` must be thread-safe, and under processes it must be picklable. ngio's side — the per-ROI readers and writers — is safe in both settings. The dask iterator surface (`iter_as_dask`, `map_as_dask`, `reduce_as_dask`) is deprecated and will be removed in ngio=1.2; for lazy whole-region access use `Image.get_as_dask` instead.

For per-ROI measurement without writing anything, use `reduce` — it returns one result per ROI, in ROI order, and takes the same `mapper` argument:

```python
means = iterator.reduce(lambda patch: float(patch.mean()))
```

## Halos: context without seams

Tiling an image and processing each tile independently leaves artifacts at the joins — a smoothing kernel at a tile edge has no neighbours to work with, and a segmentation cuts objects at the boundary. `with_halo` fixes that by reading a margin around each ROI and writing only the ROI back:

```python
iterator = iterator.by_chunks(grid="write").with_halo(x=8, y=8)
iterator.map(smooth, mapper=ThreadedMapper("auto"))
```

`smooth` receives the grown region and must return it grown too; the border is cropped off before the write, so it never lands on disk. Margins are in pixels and clip at the image borders, so an edge tile simply grows on the sides where there is room.

The ROIs themselves do not move, which is the whole point of doing this on the read side: write footprints are unchanged, so a haloed iterator parallelizes exactly as far as it did without one. Overlapping *writes* would have to be serialized; overlapping reads cost nothing.

Read the same trick backwards and it is a "trim": if you want each tile's outer margin discarded rather than written, that is exactly a halo of that width.

## Merging instead of overwriting

Writes replace what is on disk. `MergeTransform` combines with it instead — reading the destination back, folding it with your patch, and writing the result:

```python
from ngio.transforms import MergeTransform

image.set_roi(roi, patch, transforms=[MergeTransform("max")])
```

`"max"`, `"min"` and `"sum"` are order-independent, so overlapping regions give the same answer whatever order they are written in. `"keep_nonzero"` ("the last nonzero write wins") and a custom `(existing, patch, ctx) -> array` rule do depend on the order.

Such a transform must be the last in the chain, and there can be only one — the pipes raise otherwise. Both rules come from the same fact: it reads the destination through the transforms placed *before* it, so anything after would be skipped during that read, and a second one would merge an already-merged array. `MaskTransform` is the other transform of this kind, which is why it cannot be combined with a merge.

## Next steps

- [Image processing tutorial](../tutorials/image_processing.md) — an iterator applied end to end.
- [Image segmentation tutorial](../tutorials/image_segmentation.md) — segmentation and masked segmentation.
- [Iterators API reference](../api/iterators.md) — the full iterator API.
