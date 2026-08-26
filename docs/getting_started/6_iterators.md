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

<!-- Figure 14 — the five iterators, side by side -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 284" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f14t f14d">
  <title id="f14t">What each of the five iterators takes, returns and supports</title>
  <desc id="f14d">Segmentation takes an image and returns a label; masked segmentation takes an image and a mask; image processing returns an image; feature extraction and object detection are read-only and return tables. A features column lists what each one supports: every iterator takes a halo, stitching applies only to the two segmentation iterators, and the read-only ones reconcile their read margin with a join or with NMS.</desc>
  <defs>
    <g id="mci">
      <rect width="22" height="22" rx="3" fill="#151d21"></rect>
      <g fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.8" stroke-opacity=".85">
        <path transform="translate(8,8) rotate(-18) scale(0.42)" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
        <path transform="translate(15,16) rotate(12) scale(0.4)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      </g>
      <circle cx="4" cy="17" r=".7" fill="#ffffff" opacity=".5"></circle>
    </g>
    <g id="mcl">
      <rect x=".5" y=".5" width="21" height="21" rx="3" style="fill:var(--ngio-sunk);stroke:var(--ngio-line-strong)"></rect>
      <ellipse cx="8.5" cy="8" rx="6" ry="4.5" transform="rotate(-18 8.5 8)" fill="#4cae4f"></ellipse>
      <ellipse cx="15" cy="16" rx="5" ry="3.8" transform="rotate(10 15 16)" fill="#7c6bd6"></ellipse>
      <ellipse cx="5" cy="17" rx="3.2" ry="2.4" fill="#f4a63a"></ellipse>
    </g>
    <g id="mcm">
      <rect x=".5" y=".5" width="21" height="21" rx="3" style="fill:var(--ngio-sunk);stroke:var(--ngio-accent)" stroke-dasharray="3 2"></rect>
      <path transform="translate(11,11) scale(0.62)" fill="#94dad4" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    </g>
    <g id="mct">
      <rect x=".75" y=".75" width="20.5" height="20.5" rx="2.5" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
      <path d="M0 7h22M7 0v22M14 0v22" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></path>
    </g>
  </defs>

  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="16" y="34">ITERATOR</text><text x="248" y="34">IN → OUT</text><text x="368" y="34">TOPIC VERB</text><text x="472" y="34" data-comment-anchor="halo-col">FEATURES</text>
  </g>
  <g style="stroke:var(--ngio-line)"><path d="M16 44h608M16 92h608M16 140h608M16 188h608M16 236h608"></path></g>

  <g style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">
    <text x="16" y="72">SegmentationIterator</text><text x="16" y="120">MaskedSegmentationIterator</text><text x="16" y="168">ImageProcessingIterator</text><text x="16" y="216">FeatureExtractorIterator</text><text x="16" y="264">ObjectDetectionIterator</text>
  </g>

  <use href="#mci" x="248" y="57"></use>
  <use href="#mci" x="248" y="105"></use><use href="#mcm" x="272" y="105"></use>
  <use href="#mci" x="248" y="153"></use>
  <use href="#mci" x="248" y="201"></use><use href="#mcl" x="272" y="201"></use>
  <use href="#mci" x="248" y="249"></use>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M300 68h14M308 63l6 5-6 5"></path><path d="M300 116h14M308 111l6 5-6 5"></path><path d="M300 164h14M308 159l6 5-6 5"></path><path d="M300 212h14M308 207l6 5-6 5"></path><path d="M300 260h14M308 255l6 5-6 5"></path>
  </g>
  <use href="#mcl" x="322" y="57"></use><use href="#mcl" x="322" y="105"></use><use href="#mci" x="322" y="153"></use><use href="#mct" x="322" y="201"></use><use href="#mct" x="322" y="249"></use>

  <g style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color)">
    <text x="368" y="72">segment</text><text x="368" y="120">segment</text><text x="368" y="168">process</text><text x="368" y="216">measure</text><text x="368" y="264">detect</text>
  </g>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)" stroke-width="1">
    <rect x="472.5" y="59.5" width="37" height="17" rx="4"></rect><rect x="515.5" y="59.5" width="65" height="17" rx="4"></rect>
    <rect x="472.5" y="107.5" width="37" height="17" rx="4"></rect><rect x="515.5" y="107.5" width="65" height="17" rx="4"></rect>
    <rect x="472.5" y="155.5" width="37" height="17" rx="4"></rect>
    <rect x="472.5" y="203.5" width="65" height="17" rx="4"></rect><rect x="543.5" y="203.5" width="37" height="17" rx="4"></rect>
    <rect x="472.5" y="251.5" width="65" height="17" rx="4"></rect><rect x="543.5" y="251.5" width="31" height="17" rx="4"></rect>
  </g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)">
    <text x="479" y="72">halo</text><text x="522" y="72">stitching</text>
    <text x="479" y="120">halo</text><text x="522" y="120">stitching</text>
    <text x="479" y="168">halo</text>
    <text x="479" y="216">read halo</text><text x="550" y="216">join</text>
    <text x="479" y="264">read halo</text><text x="550" y="264">NMS</text>
  </g>
</svg>
</div>

- `SegmentationIterator` — `segment` an image into a label; see the [image segmentation tutorial](../tutorials/image_segmentation.md), and the [stitching tutorial](../tutorials/stitching.md) for tiling and seams.
- `MaskedSegmentationIterator` — the same, restricted to the objects of a masking ROI table (segment cells only within a tissue region, say); same tutorials.
- `ImageProcessingIterator` — `process` an image into a new image (a filter, a projection, a restoration model); see the [image processing tutorial](../tutorials/image_processing.md).
- `FeatureExtractorIterator` — read-only; `measure` joins per-region measurements into one feature table; see the [feature extraction tutorial](../tutorials/feature_extraction.md).
- `ObjectDetectionIterator` — read-only; `detect` turns a tile-by-tile detector into one deduplicated ROI table; see the [object detection tutorial](../tutorials/object_detection.md).

The verbs come in two layers. Every iterator shares the *generic* layer — `map`
(apply and write back), `reduce` (collect without writing), the hand-driven
`iter` loop, and the distributed steps `prepare_jobs`/`for_job`/`finalize` —
and each iterator adds one *topic verb* that says what the iteration means:
`segment`, `process`, `measure`, `detect`. On the writers the topic verb is
`map` under its domain name; on the read-only iterators it also runs the final
join (the feature join, the detection NMS) and returns the table. All four
are partition-aware: on a `for_job` slice they do only that job's share, and
`finalize()` is the one gather, whatever the iterator.

## Building one

Every iterator is constructed from the images it reads and writes, then narrowed. A fresh
iterator covers the whole image as a single region:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:setup"
```

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:build"
```

The constructors share a small vocabulary of keyword arguments:
`channel_selection` restricts the input reads to given channels, `axes_order`
fixes the axes order of the patches the function sees (`"yx"` above),
`input_transforms`/`output_transforms` apply
[transforms](../api/ngio/transforms.md) around the function, and the writing
iterators take `consolidation_mode` (how the output pyramid is rebuilt at the
end). The full signatures live in the
[API reference](../api/iterators.md).

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
wave, see below, and on segmentation the overlap needs a declared resolution —
`with_stitch(...)` or `on_overlap(...)`), and `"drop"` discards it.
`by_blocks(num_x=..., num_y=...)` is the complement — you say how many tiles, not how
big, and the partition is balanced by construction. `by_chunks()` tiles by the *input*
image's chunk grid, the natural unit of reading; `by_write_units()` tiles by the
*output*'s write granularity — the shard shape when the output is sharded, the chunk
shape otherwise, inspectable as `image.write_granularity` — which makes parallel
writes collision-free by construction, so a parallel `map` runs as a single
fully-parallel wave.

<!-- Figure 10 — what happens to the leftover -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 260" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f10t f10d">
  <title id="f10t">The four tail policies on a 100 pixel axis tiled at 32</title>
  <desc id="f10d">Clip shrinks the last tile to 4 pixels. Balance re-splits the last two into 18 and 18. Shift keeps every tile full size by sliding the last one back, so it overlaps its neighbour by 28 pixels — the hatched band. Drop discards the leftover entirely, leaving three tiles.</desc>
  <defs><pattern id="tphatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><path d="M0 0V6" style="stroke:var(--ngio-magenta)" stroke-width="1.4"></path></pattern></defs>

  <text x="16" y="20" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">100 PX ALONG X, TILED AT 32</text>
  <g style="stroke:var(--ngio-line-strong)" stroke-width="1"><path d="M128 46h400M128 42v8M256 42v8M384 42v8M512 42v8M528 42v8"></path></g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--md-default-fg-color--lighter)"><text x="128" y="39">0</text><text x="256" y="39">32</text><text x="384" y="39">64</text><text x="512" y="39">96</text><text x="532" y="39">100</text></g>

  <g style="font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:var(--md-default-fg-color)">
    <text x="16" y="88">tail="clip"</text><text x="16" y="136">tail="balance"</text><text x="16" y="184">tail="shift"</text><text x="16" y="232">tail="drop"</text>
  </g>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)" stroke-width="1.5">
    <rect x="128.75" y="68.75" width="126.5" height="30.5" rx="3"></rect><rect x="256.75" y="68.75" width="126.5" height="30.5" rx="3"></rect><rect x="384.75" y="68.75" width="126.5" height="30.5" rx="3"></rect><rect x="512.75" y="68.75" width="14.5" height="30.5" rx="3"></rect>
    <rect x="128.75" y="116.75" width="126.5" height="30.5" rx="3"></rect><rect x="256.75" y="116.75" width="126.5" height="30.5" rx="3"></rect><rect x="384.75" y="116.75" width="70.5" height="30.5" rx="3"></rect><rect x="456.75" y="116.75" width="70.5" height="30.5" rx="3"></rect>
    <rect x="128.75" y="164.75" width="126.5" height="30.5" rx="3"></rect><rect x="256.75" y="164.75" width="126.5" height="30.5" rx="3"></rect><rect x="384.75" y="164.75" width="126.5" height="30.5" rx="3"></rect><rect x="400.75" y="164.75" width="126.5" height="30.5" rx="3"></rect>
    <rect x="128.75" y="212.75" width="126.5" height="30.5" rx="3"></rect><rect x="256.75" y="212.75" width="126.5" height="30.5" rx="3"></rect><rect x="384.75" y="212.75" width="126.5" height="30.5" rx="3"></rect>
  </g>
  <rect x="400" y="165" width="112" height="30" fill="url(#tphatch)" opacity=".5"></rect>
  <rect x="512.75" y="212.75" width="14.5" height="30.5" rx="3" fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5" stroke-dasharray="3 3"></rect>

  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--ngio-accent-ink)">
    <text x="192" y="88">32</text><text x="320" y="88">32</text><text x="448" y="88">32</text><text x="520" y="88">4</text>
    <text x="192" y="136">32</text><text x="320" y="136">32</text><text x="420" y="136">18</text><text x="492" y="136">18</text>
    <text x="192" y="184">32</text><text x="320" y="184">32</text>
    <text x="192" y="232">32</text><text x="320" y="232">32</text><text x="448" y="232">32</text>
  </g>

  <g style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">
    <text x="544" y="88">a thin tile</text><text x="544" y="136">no thin tile</text><text x="544" y="184">two 32s overlap</text><text x="544" y="232">discarded</text>
  </g>
</svg>
</div>

Two more calls *broadcast* rather than tile: `by_yx()` splits each region into one 2D
plane per remaining coordinate (every `t`/`z`/`c` combination, full y/x extent — the
shape for "run this 2D function on every plane"), and `by_zyx(strict=...)` does the
same per 3D volume. The tutorial snippets use them wherever a 2D or 3D function meets
a higher-dimensional image.

From here you would call the topic verb (`process` here) or iterate with `iter_as_numpy` to do the work;
the [image processing tutorial](../tutorials/image_processing.md) carries this through to
a written result. (`iter_as_numpy` is `iter(data_mode="numpy")` — a bare
`iter()` still defaults to dask and warns; numpy becomes the default in
`ngio=1.2`.)

More complete examples can be found in the [Fractal tasks template](https://github.com/fractal-analytics-platform/fractal-tasks-template).

## Parallel mapping

`map` runs one ROI at a time by default, and that stays the default — parallel writing is explicit opt-in. Concurrency belongs to the *mapper*: pass one, and it sizes its own pool.

The examples from here on run on a small synthetic image — two bright blobs:

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:synthetic_setup"
```

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:mapper_demo"
```

`ThreadedMapper` is the fit for IO-bound work and for funcs that release the GIL (most numpy/scipy do); `"auto"` sizes the pool for round-trip-bound work. For pure-Python, GIL-holding funcs use `ProcessMapper(max_workers=...)` instead — the func must be picklable (a module-level function, not a lambda), and the store must not be in-memory.

Parallel writes need no locks: the mappers schedule the ROIs into
conflict-free **waves**, so no two concurrent writes ever touch the same
chunk or shard. `by_write_units()` gives a single fully-parallel wave by
construction; other tilings just run more waves.

<!-- Figure 13 — conflict-free write waves -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 344" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f13t f13d">
  <title id="f13t">How overlapping write footprints are scheduled into conflict-free waves</title>
  <desc id="f13d">The regions tile the whole output array, but they are not chunk-aligned, so every chunk is written by two neighbouring regions. Regions that share a chunk are placed in different waves: the first wave runs regions one, three and five, the second runs regions two and four. Within a wave no two writes touch the same chunk, so no locks are needed.</desc>

  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="16" y="72">CHUNK GRID</text><text x="16" y="136">REGIONS</text><text x="16" y="232">WAVE 1</text><text x="16" y="296">WAVE 2</text>
  </g>

  <g style="fill:var(--ngio-surface);stroke:var(--ngio-blue)" stroke-width="1.5">
    <rect x="112.75" y="48.75" width="118.5" height="38.5"></rect><rect x="232.75" y="48.75" width="118.5" height="38.5"></rect><rect x="352.75" y="48.75" width="118.5" height="38.5"></rect><rect x="472.75" y="48.75" width="118.5" height="38.5"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--ngio-blue-t)"><text x="172" y="73">chunk 1</text><text x="292" y="73">chunk 2</text><text x="412" y="73">chunk 3</text><text x="532" y="73">chunk 4</text></g>

  <g style="fill:var(--ngio-magenta)" fill-opacity=".1"><rect x="113" y="113" width="478" height="38"></rect></g>
  <g fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5">
    <rect x="112.75" y="112.75" width="94.5" height="38.5"></rect><rect x="208.75" y="112.75" width="94.5" height="38.5"></rect><rect x="304.75" y="112.75" width="94.5" height="38.5"></rect><rect x="400.75" y="112.75" width="94.5" height="38.5"></rect><rect x="496.75" y="112.75" width="94.5" height="38.5"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color)"><text x="160" y="137">1</text><text x="256" y="137">2</text><text x="352" y="137">3</text><text x="448" y="137">4</text><text x="544" y="137">5</text></g>
  <path d="M232 92v20M352 92v20M472 92v20" style="stroke:var(--ngio-blue)" stroke-width="1" stroke-dasharray="3 3" opacity=".7"></path>
  <text x="112" y="172" style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;fill:var(--md-default-fg-color--light)">the regions are not chunk-aligned</text>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)" stroke-width="1.5">
    <rect x="112.75" y="208.75" width="94.5" height="38.5"></rect><rect x="304.75" y="208.75" width="94.5" height="38.5"></rect><rect x="496.75" y="208.75" width="94.5" height="38.5"></rect>
  </g>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.2" stroke-dasharray="3 3">
    <rect x="208.75" y="208.75" width="94.5" height="38.5"></rect><rect x="400.75" y="208.75" width="94.5" height="38.5"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--ngio-accent-ink)"><text x="160" y="233">1</text><text x="352" y="233">3</text><text x="544" y="233">5</text></g>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)" stroke-width="1.5">
    <rect x="208.75" y="272.75" width="94.5" height="38.5"></rect><rect x="400.75" y="272.75" width="94.5" height="38.5"></rect>
  </g>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.2" stroke-dasharray="3 3">
    <rect x="112.75" y="272.75" width="94.5" height="38.5"></rect><rect x="304.75" y="272.75" width="94.5" height="38.5"></rect><rect x="496.75" y="272.75" width="94.5" height="38.5"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--ngio-accent-ink)"><text x="256" y="297">2</text><text x="448" y="297">4</text></g>

  <text x="16" y="332" style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color--light)">Every region in a wave runs in parallel; waves run one after another.</text>
</svg>
</div>

??? note "Wave order is the canonical write order"
    The serial mappers run the same wave order as the parallel ones — every
    mapper writes the same bytes, and two runs of the same version are
    bit-identical. For ROIs whose *pixels* genuinely overlap (a `by_grid`
    stride below the size, a `"shift"` tail), which write wins the shared
    pixels is schedule-defined by default — performance first; declare
    `write_order="roi"` for the reproducible later-ROI-wins order (see the
    next note). Mask-protected and pixel-disjoint writes are order-independent
    either way, as is an order-independent `merge=` (`"max"`, `"min"`,
    `"sum"`).

??? note "Reproducible seams: `write_order="roi"`"
    By default contested writes are scheduled for parallelism alone
    (`write_order="any"`): exactness always — no two overlapping writes
    ever run concurrently, merges always apply, stitch unification is
    unchanged — and a run is deterministic per ngio version (the schedule
    is a pure function of the ROI list, identical under every mapper). What
    is schedule-defined is *which tile owns a contested pixel*: it can
    differ from the manual `iter` loop and can change across ngio versions
    or retilings — pixel-visible differences, not rounding noise.
    Order-independent merges (`"max"`, `"min"`, `"sum"`, `"keep_nonzero"`)
    write identical pixels regardless, so for them the default costs
    nothing.

    When seam ownership must be reproducible — the later ROI wins, `map`
    bit-identical to the hand-driven loops, stable across versions and
    mappers — opt in per declaration: `on_overlap(..., write_order="roi")`
    or `StitchConfig(write_order="roi")`. The price is parallelism on
    genuinely overlapping tilings: an overlapping grid becomes a chain of
    ordered neighbours, measured 1.5–2.5× slower on parallel mappers
    (serial runs are unaffected). Disjoint tilings (`by_write_units()`,
    halo cores, chunk grids) are bit-identical under either value. A pair
    is relaxed only when *both* sides declare `"any"` — one `"roi"`
    declaration keeps its ordering against everything.

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

This is the one mapper that changes the func's contract: it receives a stacked batch with a leading batch axis and must return an array with the same leading axis. Everything else composes as usual — tilings are ragged in general (border tiles clip, halos shrink at image edges, masked regions are arbitrary), so each batch is padded up to its per-axis maximum before stacking (`pad_mode`/`pad_values` choose how) and every output is sliced back to its region's true shape before the write; a halo is trimmed after that, exactly as under any other mapper; and `for_job` partitions batch their own share.

Within a batch the reads fan out on a thread pool (`read_workers`), while the writes run serially on the calling thread. Batches are cut over the same canonical (wave) order as every other mapper, so contested pixels land identically — and the serial writes make batched mapping write-safe on any tiling.

Prefer to drive the loop yourself? `iter(batch_size=...)` yields `(patches, writers)` — two aligned lists of up to `batch_size` items, in ROI order — and leaves the stacking (and any ragged-tile policy) to you. The run finalizes when the loop completes, exactly like the unbatched `iter`; on the read-only iterators it yields the payload lists alone (`(image, label, roi)` tuples for features, `(patch, roi)` for detection):

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:iter_batched_demo"
```

## Distributed runs

A mapper parallelizes within one machine. On a cluster — SLURM array tasks over a shared filesystem, where processes cannot coordinate — restrict each task to one partition of the work instead:

```python
n_jobs = 4
job_index = 0  # e.g. int($SLURM_ARRAY_TASK_ID)

iterator = SegmentationIterator(image, label, ...).by_chunks()
iterator.for_job(job_index, n_jobs=n_jobs).segment(func)
```

and, in a dependent job once every array task has finished, the gather step:

```python
iterator = SegmentationIterator(image, label, ...).by_chunks()  # same construction
iterator.finalize()
```

`for_job` is a builder call like `by_grid` or `with_halo` — it returns a new iterator restricted to that partition's regions, and everything else reads as usual, `segment(func, mapper=...)` included. It comes *last* in the chain (reshaping a restricted iterator refuses), and a slice's `segment` (or `map`) deliberately does **not** finalize: the pyramid resolve is the one global step, and it belongs to the single gather job — which, thanks to region-scoped consolidation, rebuilds only what the jobs wrote. Until the gather runs, only the iterated level is up to date. (That also makes `for_job(0, 1)` the sanctioned way to *defer* a finalize on purpose: one job that writes everything, gathered whenever you choose.)

Each job builds the identical iterator — construction is metadata-only and deterministic, so this is cheap — and derives the same partition on its own; there is nothing to hand from one job to another. Partitions never share a write unit, so the jobs need no locks and no coordination, in any order and any overlap in time; regions whose footprints conflict simply travel in the same partition, where the ordinary wave planning handles them.

Effective parallelism therefore equals the number of independent groups, which follows the **output's** chunking. Inspect it before submitting: `[it.for_job(i, n_jobs=n).partition_indices for i in range(n)]`. One fat list plus empties means the output chunking (or a tiling like `by_zyx`, which splits along t only) is the constraint, not the cluster — a single-chunk output is one group by construction, since a chunk is one atomic write object. Surplus partitions are harmless no-ops, and [`write_conflict_components`][ngio.iterators.write_conflict_components] makes the grouping auditable.

The requirements mirror the model: every job must use the same `n_jobs` and the same iterator construction; the store must not be in-memory (each process would write its own private copy). The read-only iterators partition too — their global join cannot be reproduced piecewise (greedy NMS is not hierarchical: suppressing per job and then merging can keep different boxes than one global pass), so on a slice `measure`/`detect` bank a *partial* instead of joining, and `finalize()` runs the one global join and returns the table. A declared join on a slice is inert — the slice banks regardless, and the gather runs it. `finalize` refuses a half-finished run (a missing job errors instead of producing a plausible-looking, silently incomplete table), refuses on a `for_job` slice (the gather is global), and refuses when nothing was prepared or banked.

Schedulers like Fractal run distributed work as **init → parallel tasks →
consolidate**, and `prepare_jobs` is the init step: it performs any setup the
run needs — always wiping stale scratch state from earlier runs first — and
returns the *parallelization list* (one JSON-ready argument set per non-empty
partition, splatting into `for_job(**args)`). It is optional for a plain
writer — the two-step recipe above works on its own — and **required** for a
stitching segmentation (the scratch band arrays must exist, race-free, before
any job banks into them) and for the read-only iterators. The
[distributed processing tutorial](../tutorials/distributed_processing.md) makes
the whole three-phase recipe executable — partition layouts, distributed
stitching, and distributed measurement.

## Halos: context without seams

Tiling an image and processing each tile independently leaves artifacts at the joins — a smoothing kernel at a tile edge has no neighbours to work with, and a segmentation cuts objects at the boundary. `with_halo` fixes that by reading a margin around each ROI and writing only the ROI back:

<!-- Figure 07 — read grown, write the core -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 220" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f7t f7d">
  <title id="f7t">How a halo reads a grown region and writes only the core</title>
  <desc id="f7d">The region is grown by the halo margin on the read, the function sees the grown patch, and the margin is cropped off before the write — so the written region is exactly the region you asked for. Margins clip at the image border.</desc>
  <defs>
    <filter id="hgr" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch"></feTurbulence><feColorMatrix type="saturate" values="0"></feColorMatrix></filter>
    <clipPath id="hp1"><rect x="16" y="44" width="168" height="144" rx="3"></rect></clipPath>
    <clipPath id="hp2"><rect x="236" y="44" width="168" height="144" rx="3"></rect></clipPath>
    <clipPath id="hp3"><rect x="456" y="44" width="168" height="144" rx="3"></rect></clipPath>
    <clipPath id="hpc"><rect x="516" y="84" width="64" height="48"></rect></clipPath>
    <g id="hcells" fill="#c7d3d7" stroke="#f2f8f9" stroke-width="1.4" stroke-opacity=".9">
      <path transform="translate(24,20) rotate(15) scale(0.8)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      <path transform="translate(58,14) rotate(-20) scale(0.7)" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(96,22) rotate(10) scale(0.9)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(136,18) rotate(30) scale(0.75)" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(20,52) rotate(-35) scale(0.8)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(52,44) rotate(-12) scale(0.95)" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
      <path transform="translate(92,54) rotate(22) scale(0.85)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      <path transform="translate(128,50) rotate(-18) scale(0.9)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(78,50) rotate(-24) scale(0.7)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(110,66) rotate(12) scale(0.65)" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(82,74) rotate(30) scale(0.75)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      <path transform="translate(28,92) rotate(18) scale(0.9)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      <path transform="translate(64,84) rotate(-8) scale(0.75)" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(100,96) rotate(-24) scale(0.95)" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
      <path transform="translate(136,88) rotate(14) scale(0.8)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(46,124) rotate(28) scale(0.85)" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(88,128) rotate(-16) scale(0.75)" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(124,126) rotate(8) scale(0.9)" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    </g>
    <g id="hspk" fill="#ffffff">
      <circle cx="14" cy="70" r=".7" opacity=".4"></circle><circle cx="150" cy="36" r=".6" opacity=".3"></circle><circle cx="78" cy="112" r=".7" opacity=".3"></circle><circle cx="112" cy="34" r=".5" opacity=".3"></circle><circle cx="40" cy="136" r=".6" opacity=".3"></circle><circle cx="158" cy="108" r=".6" opacity=".3"></circle>
    </g>
    <g id="hlabs">
      <path transform="translate(52,44) rotate(-12) scale(0.95)" fill="#4cae4f" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
      <path transform="translate(92,54) rotate(22) scale(0.85)" fill="#f4a63a" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
      <path transform="translate(64,84) rotate(-8) scale(0.75)" fill="#22a699" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(100,96) rotate(-24) scale(0.95)" fill="#7c6bd6" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
      <path transform="translate(128,50) rotate(-18) scale(0.9)" fill="#7c6bd6" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(78,50) rotate(-24) scale(0.7)" fill="#7c6bd6" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
      <path transform="translate(110,66) rotate(12) scale(0.65)" fill="#4cae4f" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
      <path transform="translate(82,74) rotate(30) scale(0.75)" fill="#ef6f9b" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    </g>
  </defs>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)">
    <rect x="16.5" y="16.5" width="15" height="14" rx="3"></rect>
    <rect x="236.5" y="16.5" width="15" height="14" rx="3"></rect>
    <rect x="456.5" y="16.5" width="15" height="14" rx="3"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)"><text x="24" y="27">1</text><text x="244" y="27">2</text><text x="464" y="27">3</text></g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="38" y="27">THE ROI</text><text x="258" y="27">THE READ</text><text x="478" y="27">THE WRITE</text>
  </g>

  <g clip-path="url(#hp1)">
    <rect x="16" y="44" width="168" height="144" fill="#151d21"></rect>
    <use href="#hcells" transform="translate(16,44)"></use>
    <use href="#hspk" transform="translate(16,44)"></use>
    <rect x="16" y="44" width="168" height="144" filter="url(#hgr)" opacity=".1" style="mix-blend-mode:screen"></rect>
  </g>
  <rect x="76" y="84" width="64" height="48" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2.5"></rect>

  <g clip-path="url(#hp2)">
    <rect x="236" y="44" width="168" height="144" fill="#151d21"></rect>
    <use href="#hcells" transform="translate(236,44)"></use>
    <use href="#hspk" transform="translate(236,44)"></use>
    <rect x="236" y="44" width="168" height="144" filter="url(#hgr)" opacity=".1" style="mix-blend-mode:screen"></rect>
    <path d="M284 72h88v72h-88z M296 84h64v48h-64z" fill-rule="evenodd" style="fill:var(--ngio-accent)" opacity=".38"></path>
  </g>
  <rect x="284" y="72" width="88" height="72" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5" stroke-dasharray="4 3"></rect>
  <rect x="296" y="84" width="64" height="48" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2"></rect>

  <g clip-path="url(#hp3)">
    <rect x="456" y="44" width="168" height="144" fill="#151d21"></rect>
    <use href="#hcells" transform="translate(456,44)"></use>
    <rect x="516" y="84" width="64" height="48" fill="#151d21"></rect>
    <g clip-path="url(#hpc)"><use href="#hlabs" transform="translate(456,44)"></use></g>
    <rect x="456" y="44" width="168" height="144" filter="url(#hgr)" opacity=".1" style="mix-blend-mode:screen"></rect>
  </g>
  <rect x="504" y="72" width="88" height="72" fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5" stroke-dasharray="3 4"></rect>
  <rect x="516" y="84" width="64" height="48" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="2.5"></rect>

  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color--light)">
    <text x="16" y="206">base ROI</text><text x="236" y="206" style="fill:var(--ngio-accent-ink)">with_halo(x=12, y=12)</text><text x="456" y="206">halo stripped on write</text>
  </g>
</svg>
</div>

```python exec="true" source="material-block" session="iterators"
--8<-- "docs/snippets/getting_started/iterators.py:halo_demo"
```

`smooth` receives the grown region and must return it grown too; the border is cropped off before the write, so it never lands on disk. Margins are in pixels and clip at the image borders, so an edge tile simply grows on the sides where there is room.

Two read-only iterators take a halo too, as a pure read margin — there is no write to crop it from, so the overlap must be reconciled after the fact. Detection reconciles it itself: NMS removes the duplicate boxes. Feature extraction delegates it to you: patches *and* the `roi` argument cover the grown region, a border object is measured by every region that sees it, and the resulting duplicate `label` rows are yours to reconcile in a declared join (`with_join`) — every row carries `roi_index`/`roi_name` for exactly that (the default join keeps the duplicates as-is).

The ROIs themselves do not move, which is the whole point of doing this on the read side: write footprints are unchanged, so a haloed iterator parallelizes exactly as far as it did without one. Overlapping *writes* would have to be serialized; overlapping reads cost nothing.

Read the same trick backwards and it is a "trim": if you want each tile's outer margin discarded rather than written, that is exactly a halo of that width.

## Overlap and reconciliation

Every iterator has exactly one *reconciliation declaration* — the chain call
that says how per-region results become one consistent answer — and each is
backed by a swappable protocol, so a custom implementation drops in without
touching internals:

| Iterator | Declaration | Required? | Behind it |
|---|---|---|---|
| `ImageProcessing` | `on_overlap(policy)` | optional — undeclared overlap is last-writer-wins in schedule order (`write_order="roi"` for reproducible seams) | the write path's `merge=` policies (`"max"`, `"sum"`, a callable, …) |
| `Segmentation` | `with_stitch(config)` *or* `on_overlap(policy)` | **required when write footprints overlap** — undeclared overlapping label writes refuse loudly | `SeamMatcherProtocol` (`StitchConfig(seam_matcher=...)`, `IouSeamMatcher` default) |
| `MaskedSegmentation` | `with_stitch(config)` (within a mask) | never — mask-protected writes cannot contest, `on_overlap` is refused | same |
| `FeatureExtractor` | `with_join(join)` | optional — the default join keeps duplicate rows, provenance columns attached | `JoinProtocol` (`ConcatJoin` default) |
| `ObjectDetection` | `with_nms(nms)` | optional — `GreedyNms()` by default | `NmsProtocol` (`GreedyNms` default) |

The segmentation rule is the one hard requirement, and it is deliberate:
last-writer-wins on two overlapping *label* writes produces torn objects —
deterministic, but almost never the intent — so the writing verbs refuse until
you say what should happen. The check is pixel-exact on the write footprints:
a halo never triggers it (the margin is cropped before the write), and tiles
that merely share a chunk without sharing pixels pass. `on_overlap("last")`
declares exactly the old behavior; any merge rule combines with what is on
disk instead.

## Stitching a tiled segmentation

Segmenting tile by tile leaves an object that crosses a boundary as two objects with two ids — and, because every tile numbers its objects from 1, leaves ids that mean nothing outside their own tile. `with_stitch()` fixes both:

<!-- Figure 08 — overlap is the evidence, not adjacency -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 408" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f8t f8d">
  <title id="f8t">How stitching merges objects split across tile boundaries</title>
  <desc id="f8d">Each tile predicts its own objects, numbered from one within that tile. Where two tiles predicted the same pixels and their objects agree above the IoU threshold, a union-find joins the two ids into one object; objects that merely abut across a cut are left as two.</desc>
  <defs>
    <clipPath id="sca"><rect x="116" y="60" width="240" height="152" rx="7"></rect></clipPath>
    <clipPath id="scb"><rect x="284" y="60" width="240" height="152" rx="7"></rect></clipPath>
    <clipPath id="scov"><rect x="284" y="60" width="72" height="152"></rect></clipPath>
  </defs>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)"><rect x="16.5" y="8.5" width="15" height="14" rx="3"></rect></g>
  <text x="24" y="19" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)">1</text>
  <text x="38" y="19" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">EACH TILE IS SEGMENTED ALONE, WITH ITS OWN IDS</text>

  <rect x="16.5" y="32.5" width="607" height="200" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <g style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color)"><text x="116" y="52">tile A</text><text x="524" y="52" text-anchor="end">tile B</text></g>

  <g clip-path="url(#sca)">
    <rect x="116" y="60" width="240" height="152" style="fill:var(--ngio-surface)"></rect>
    <path transform="translate(144,90) rotate(-12) scale(0.8)" fill="#4cae4f" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(184,74) rotate(24) scale(1)" fill="#7c6bd6" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(220,100) rotate(-18) scale(0.8)" fill="#f4a63a" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(154,132) rotate(14) scale(0.85)" fill="#22a699" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(200,152) rotate(-8) scale(0.9)" fill="#ef6f9b" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(252,84) rotate(30) scale(1)" fill="#f4a63a" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(256,140) rotate(-22) scale(0.75)" fill="#4cae4f" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
    <path transform="translate(270,182) rotate(10) scale(0.8)" fill="#7c6bd6" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(172,190) rotate(-14) scale(1)" fill="#22a699" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(218,196) rotate(20) scale(0.7)" fill="#4cae4f" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(130,164) rotate(-26) scale(0.9)" fill="#ef6f9b" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(310,124) rotate(-6) scale(0.9)" fill="#f4a63a" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
  </g>
  <g clip-path="url(#scb)">
    <rect x="284" y="60" width="240" height="152" style="fill:var(--ngio-surface)"></rect>
    <path transform="translate(315,127) rotate(-6) scale(0.9)" fill="#ef6f9b" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
    <path transform="translate(356,84) rotate(18) scale(0.85)" fill="#22a699" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(400,72) rotate(-24) scale(1)" fill="#4cae4f" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(376,162) rotate(12) scale(0.8)" fill="#7c6bd6" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(404,140) rotate(-16) scale(0.8)" fill="#f4a63a" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
    <path transform="translate(456,96) rotate(26) scale(0.85)" fill="#ef6f9b" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(448,168) rotate(-10) scale(0.85)" fill="#4cae4f" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(496,120) rotate(14) scale(1)" fill="#22a699" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(380,196) rotate(-20) scale(0.8)" fill="#f4a63a" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
    <path transform="translate(434,204) rotate(22) scale(0.9)" fill="#7c6bd6" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
    <path transform="translate(504,180) rotate(-14) scale(0.8)" fill="#7c6bd6" d="M-11-2c-.4-5.4 4.4-8.6 10.6-8.2 6.8.4 11.2 4.2 11.2 9.6 0 5-3.4 8.6-8.8 9.2-4.6.6-9-.4-11.4-3-1.2-1.4-1.6-3.6-1.6-7.6Z"></path>
    <path transform="translate(490,66) rotate(8) scale(0.9)" fill="#4cae4f" d="M-8-1.2c-.2-3.8 3-6.2 7.2-5.8 4.4.4 8 3 8.4 6.6.4 3.8-2.8 7.2-7.4 7.4C-4 7.2-7.4 4.6-8-1.2Z"></path>
  </g>
  <g clip-path="url(#scov)">
    <path transform="translate(310,124) rotate(-6) scale(0.9)" fill="#f4a63a" opacity=".9" d="M-18-3c.4-5.4 7-7.4 15-6.8 9 .6 18 2.8 21.6 6.4 1.8 1.8 1 5-2 7.8-4.2 4-13.2 7.2-21.2 6.2C-12 9.6-17 6-18-3Z"></path>
  </g>
  <rect x="284" y="60" width="72" height="152" style="fill:var(--ngio-accent)" opacity=".2"></rect>
  <path transform="translate(302,178) rotate(0) scale(1)" fill="#4cae4f" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <path transform="translate(327,178) rotate(180) scale(1)" fill="#7c6bd6" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>

  <g fill="none" stroke-width="1.5">
    <rect x="116.75" y="60.75" width="238.5" height="150.5" rx="7" style="stroke:var(--ngio-magenta)"></rect>
    <rect x="284.75" y="60.75" width="238.5" height="150.5" rx="7" style="stroke:var(--ngio-magenta)"></rect>
  </g>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)"><rect x="16.5" y="256.5" width="15" height="14" rx="3"></rect></g>
  <text x="24" y="267" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)">2</text>
  <text x="38" y="267" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">UNION-FIND DECIDES WHICH PAIRS ARE ONE OBJECT</text>

  <rect x="16.5" y="280.5" width="295" height="112" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <path opacity=".85" transform="translate(106,321) rotate(-6) scale(1.4)" fill="#f4a63a" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <path opacity=".7" transform="translate(113,325) rotate(-6) scale(1.4)" fill="#ef6f9b" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5"><path d="M152 323h32M178 318l6 5-6 5"></path></g>
  <text x="168" y="363" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color)">IoU 0.78 ≥ 0.5</text>
  <path transform="translate(222,323) rotate(-6) scale(1.4)" fill="#f4a63a" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>

  <rect x="328.5" y="280.5" width="295" height="112" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
  <path transform="translate(387,323) rotate(0) scale(1.4)" fill="#4cae4f" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <path transform="translate(422,323) rotate(180) scale(1.4)" fill="#7c6bd6" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <path d="M405 306V340" style="stroke:var(--ngio-line-strong)" stroke-width="1.5" stroke-dasharray="3 4"></path>
  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5"><path d="M460 323h32M486 318l6 5-6 5"></path></g>
  <text x="476" y="363" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color)">IoU 0 &lt; 0.5</text>
  <path transform="translate(532,323) rotate(0) scale(1.4)" fill="#4cae4f" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
  <path transform="translate(567,323) rotate(180) scale(1.4)" fill="#7c6bd6" d="M-13-2.2c-.4-4.6 4.4-7.2 11-6.8 7.4.4 13.8 3.4 15.4 7.2 1 2.4-1.4 5.4-6 7.4-5.8 2.4-12.8 2-17-.4-2.4-1.4-3.4-3.8-3.4-7.4Z"></path>
</svg>
</div>

Any ROI list stitches: a regular grid with a halo, an overlapping-FOV microscope layout, a ragged ROI table. The criterion is that two tiles' predictions **overlap**, not that their objects merely touch across a cut — two distinct objects that abut at a boundary are adjacent but do not overlap, so an adjacency rule would merge them and the overlap rule does not. The shared opinion comes from a halo (each tile reads past its own edge), from the tiles genuinely overlapping (FOV layouts need no halo — the overlap *is* the evidence), or both; with neither, stitching refuses, since no two tiles ever predict the same pixel.

During the map each tile banks its grown prediction into transient per-tile scratch arrays, removed once the stitch resolves. Where two tiles wrote the same *output* pixels and their objects did not match, the schedule picks the owner (unification itself is exact and order-free); `StitchConfig(write_order="roi")` makes those seams deterministically owned by the later ROI instead.

`MaskedSegmentationIterator` takes `with_stitch()` too, for tiling *within* a mask: a huge masked object tiled with `by_grid` + `with_halo` gets its split sub-objects merged, each tile banks only what its own mask can write, and tiles of different masks are never compared — an object cannot span two masks. Ids come out unique and dense across every object, so no `UniqueLabelsTransform` is needed (combining it with `stitch` raises).

The [stitching tutorial](../tutorials/stitching.md) walks the whole flow on a
real image — the naive per-tile route, `with_stitch()`, and the `StitchConfig`
tuning knobs (`iou_threshold`, `block_size`, `scratch_store`).

## Detecting objects into a ROI table

Not every model produces a mask. An object detector — a YOLO network, a spot finder — reports **bounding boxes**, and the natural home for those is a ROI table, not a label image. The `ObjectDetectionIterator` runs a detector tile by tile and returns one `RoiTable` of the objects it found:

<!-- Figure 11 — tile-by-tile detection into one ROI table -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 256" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f11t f11d">
  <title id="f11t">How the object detection iterator turns per-tile boxes into one deduplicated ROI table</title>
  <desc id="f11d">Each tile reads a halo past its edge so a spot at the boundary is seen whole by at least one tile. Both neighbours then report it, so non-maximum suppression keeps the higher-scoring box, and the survivors are merged into one ROI table in world coordinates.</desc>
  <defs>
    <clipPath id="odf"><rect x="16" y="64" width="176" height="176" rx="3"></rect></clipPath>
  </defs>

  <g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)">
    <rect x="16.5" y="24.5" width="15" height="14" rx="3"></rect><rect x="232.5" y="24.5" width="15" height="14" rx="3"></rect><rect x="448.5" y="24.5" width="15" height="14" rx="3"></rect>
  </g>
  <g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)"><text x="24" y="35">1</text><text x="240" y="35">2</text><text x="456" y="35">3</text></g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="38" y="35">DETECT PER TILE</text><text x="254" y="35">DEDUPLICATE</text><text x="470" y="35">MERGE</text>
  </g>

  <g clip-path="url(#odf)">
    <rect x="16" y="64" width="176" height="176" fill="#151d21"></rect>
    <g fill="#e8f0f2"><circle cx="48" cy="100" r="4"></circle><circle cx="150" cy="96" r="3.4"></circle><circle cx="104" cy="152" r="4.4"></circle><circle cx="60" cy="196" r="3.6"></circle><circle cx="160" cy="204" r="3"></circle><circle cx="76" cy="80" r="3.2"></circle><circle cx="132" cy="130" r="3.8"></circle><circle cx="36" cy="160" r="3"></circle><circle cx="176" cy="144" r="3.4"></circle><circle cx="116" cy="222" r="3.6"></circle><circle cx="72" cy="128" r="2.8"></circle></g>
    <g fill="#ffffff" opacity=".3"><circle cx="30" cy="182" r=".7"></circle><circle cx="140" cy="164" r=".6"></circle><circle cx="182" cy="112" r=".7"></circle><circle cx="90" cy="212" r=".6"></circle></g>
    <path d="M104 64V240M16 152H192" stroke="#ffffff" stroke-width="1.2" opacity=".35"></path>
  </g>
  <rect x="16.5" y="64.5" width="175" height="175" rx="3" fill="none" style="stroke:var(--ngio-line-strong)"></rect>
  <rect x="80" y="128" width="48" height="48" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.2" stroke-dasharray="4 3"></rect>
  <g fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5">
    <rect x="37.5" y="89.5" width="21" height="21"></rect><rect x="139.5" y="85.5" width="21" height="21"></rect>
    <rect x="91.5" y="139.5" width="25" height="25"></rect><rect x="95.5" y="143.5" width="25" height="25"></rect>
    <rect x="49.5" y="185.5" width="21" height="21"></rect><rect x="149.5" y="193.5" width="21" height="21"></rect>
    <rect x="65.5" y="69.5" width="21" height="21"></rect><rect x="121.5" y="119.5" width="21" height="21"></rect>
    <rect x="25.5" y="149.5" width="21" height="21"></rect><rect x="165.5" y="133.5" width="21" height="21"></rect>
    <rect x="105.5" y="211.5" width="21" height="21"></rect><rect x="61.5" y="117.5" width="21" height="21"></rect>
  </g>

  <g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5"><path d="M200 152h24M218 147l6 5-6 5"></path><path d="M416 152h24M434 147l6 5-6 5"></path></g>

  <rect x="232.5" y="64.5" width="175" height="175" rx="3" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5"></rect>
  <text x="246" y="92" style="font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;fill:var(--md-default-fg-color)">non-maximum suppression</text>
  <g fill="none" stroke-width="1.5"><rect x="272.5" y="128.5" width="41" height="41" style="stroke:var(--ngio-magenta)"></rect><rect x="286.5" y="142.5" width="41" height="41" style="stroke:var(--ngio-line-strong)" stroke-dasharray="3 3"></rect></g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px"><text x="322" y="124" style="fill:var(--ngio-magenta-t)">0.91</text><text x="334" y="196" style="fill:var(--md-default-fg-color--lighter)">0.68</text></g>
  <text x="320" y="220" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:11px;fill:var(--md-default-fg-color--light)">iou ≥ 0.5</text>

  <rect x="448.5" y="64.5" width="175" height="175" rx="3" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
  <path d="M448 92h176M504 64V240M560 64V240M448 122h176M448 152h176M448 182h176M448 212h176" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></path>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--md-default-fg-color--light)"><text x="458" y="84">label</text><text x="514" y="84">x, y</text><text x="570" y="84">conf</text></g>
  <g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:var(--md-default-fg-color)">
    <text x="458" y="112">1</text><text x="458" y="142">2</text><text x="458" y="172">3</text><text x="458" y="202">4</text><text x="458" y="232">…</text>
    <text x="514" y="112">world</text><text x="514" y="142">world</text><text x="514" y="172">world</text><text x="514" y="202">world</text><text x="514" y="232">…</text>
    <text x="570" y="112">0.94</text><text x="570" y="142">0.91</text><text x="570" y="172">0.88</text><text x="570" y="202">0.82</text><text x="570" y="232">…</text>
  </g>
</svg>
</div>

NMS is declared with `with_nms(GreedyNms(iou_threshold=..., score_column=...))`, exactly as stitching is with `with_stitch(StitchConfig(...))` — and both defaults are swappable protocols: any object with `score_column`, `max_detections_per_tile`, and a deterministic `suppress(detections)` satisfies `NmsProtocol` (soft-NMS, class-aware suppression), and a `StitchConfig(seam_matcher=...)` replaces the IoU criterion with your own `(patch_a, patch_b) -> [(id_a, id_b), ...]` pair decision. A parallel `mapper=` on `detect` fans the tiles out like any `reduce`.

The detector sees one tile at a time and answers in the tile's own pixels; the iterator does the bookkeeping the detector should not — anchoring each tile's boxes into the reference image's world coordinates, and resolving the boundary problem.

!!! note "Box contract"
    - `(patch) -> list[Roi]`, each box in the tile's own pixels:
      `Roi.from_values(slices={"x": (x0, width), "y": (y0, height)}, name=None, space="pixel", confidence=0.9)`.
      `space="pixel"` is required; a world-space box is refused — patch-local
      numbers in a world-labelled Roi would land every box in the wrong place
      silently.
    - Boxes pin `x` and `y` (optionally `z`), and every tile must report the
      same dimensionality, scored or unscored — mixtures raise.
    - Extra fields (confidence, class) ride into the table unchanged; `name`
      and `label` are refused — the iterator assigns them when it renumbers
      the survivors (put a class in an extra field, e.g. `class_id`).
    - A 2D detector on a 3D or timelapse image is deduplicated only within
      each z-slab or time point; boxes keep their tile's extent along the
      axes they do not pin.

The boundary problem is the sliding-window one. An object cut by a tile edge is seen only partially by either tile, so each tile reads a halo past its edge — on a read-only iterator the halo is a pure read margin, there being no write to crop it from — and the object is seen whole by at least one of them. The cost is that both neighbours now report it, and the cure is standard **non-maximum suppression**: boxes overlapping at or above `iou_threshold` (default `0.5`) are one object, and the one ranked higher by the `score_column` (`"confidence"` by default; box volume when the detector reports no score) survives. Per-tile NMS inside the detector composes cleanly with this cross-tile pass. The survivors are renumbered to a dense `1..N` and returned; like `measure`, nothing is written — storing the table is your `add_table` call.

The [object detection tutorial](../tutorials/object_detection.md) runs a spot
finder through `detect` end to end, shows the suppression happening on the raw
pre-NMS boxes, and covers `Roi.anchor` — the coordinate call you can reuse in
a custom flow.

## What each iterator supports

| | `ImageProcessing` | `Segmentation` | `MaskedSegmentation` | `FeatureExtractor` | `ObjectDetection` |
|---|---|---|---|---|---|
| Writes to | image | label | label (masked) | — | — |
| Topic verb | `process` [5] | `segment` [5] | `segment` [5] | `measure` | `detect` |
| `reduce` | ✓ | ✓ | ✓ | ⚠ [6] | ⚠ [1] |
| `with_halo` | ✓ | ✓ | ✓ | ✓ [6] | ✓ [1] |
| `with_stitch` | — | ✓ [2] | ✓ within a mask [2] | — | — |
| Overlapping ROIs | ✓ [3] | ✓ [3] | ✓ [3] | ✓ | ✓ [1] |
| `ThreadedMapper` / `ProcessMapper` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BatchedMapper` | ✓ | ✓ | ✓ | ✗ [4] | ✗ [4] |
| Distributed gather | `finalize` | `finalize` | `finalize` | `finalize` → table | `finalize` → table |

1. Detection reconciles overlap itself: the halo is a pure read margin and NMS removes the duplicate boxes — but only through `detect`. `reduce` returns raw per-tile boxes.
2. Stitching takes any ROI list. On the masked iterator it merges sub-objects split by a tile boundary *within one mask*; tiles of different masks are never compared, and ids come out unique across every object — no `UniqueLabelsTransform` needed (combining it with `stitch` raises).
3. Overlapping writes are safe under every mapper: they are wave-scheduled and run in the same order everywhere, so a run is deterministic per version. Which write wins a contested pixel is schedule-defined by default; `write_order="roi"` makes it the later ROI and `map` bit-identical to the manual `iter` loop. Masked writes never contest — each touches only its own object's pixels. A commutative `merge=` (`"max"`, `"sum"`) removes the order-dependence outright. One configuration to avoid: an *in-place* run (same array in and out) with an overlapping tiling — reads then race the neighbouring writes; use a separate output (a halo makes this refuse outright).
4. `BatchedMapper` stacks plain arrays; these iterators hand `func` tuple payloads.
5. On the writers the topic verb is the generic `map` under its domain name; both remain available.
6. The feature halo is a pure read margin: patches and the `roi` argument grow, a border object can be measured by several regions, and the duplicate rows are yours to reconcile in a declared join (`with_join`) via the stamped `roi_index`/`roi_name` (the default join keeps them, silently). `reduce`/`iter` read the grown regions too. Without a halo, `reduce` is unrestricted.

Restrictions that hold everywhere:

- `with_stitch` needs a halo or overlapping ROIs, stays on the numpy path, and cannot combine with `on_overlap` (the stitch owns the contested pixels).
- A haloed *writer* refuses `reduce` and read-only `iter`; an in-place run (same array in and out) refuses a halo.
- `by_write_units` on the read-only iterators falls back to the input's chunks.
- The deprecated dask verbs run serially, and `ProcessMapper` refuses in-memory stores.

## Next steps

- [Image processing tutorial](../tutorials/image_processing.md) — an iterator applied end to end.
- [Image segmentation tutorial](../tutorials/image_segmentation.md) — segmentation and masked segmentation.
- [Stitching tutorial](../tutorials/stitching.md) — a tiled segmentation from the naive route to a tuned `with_stitch()`.
- [Object detection tutorial](../tutorials/object_detection.md) — a spot finder through `detect`, with the NMS walkthrough.
- [Feature extraction tutorial](../tutorials/feature_extraction.md) — `measure` on a segmented image.
- [Distributed processing tutorial](../tutorials/distributed_processing.md) — the three-phase recipe end to end.
- [Iterators API reference](../api/iterators.md) — the full iterator API.
