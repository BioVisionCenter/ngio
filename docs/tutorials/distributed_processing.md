---
description: Split one iterator run across cluster jobs with prepare_jobs, for_job and finalize.
---

# Distributed processing

**Split one iterator run across jobs that never talk to each other.**

On a cluster, an iterator run is split into *jobs* — SLURM array tasks,
[Fractal](https://fractal-analytics-platform.github.io/) parallel tasks — that share a
filesystem but cannot talk to each other. ngio's
[partition model](../getting_started/6_iterators.md#distributed-runs) is built for
exactly that: each job restricts the same iterator to its own share with `for_job`,
and no locks or coordination are ever needed. This tutorial makes the recipe run on a
laptop by standing a plain Python loop in for the cluster — one loop iteration per
job, where a job is whatever one scheduler task would run.

## Step 1: set up

```python exec="true" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:plot_helpers"
```

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:open_container"
```

The segmentation function is the watershed pipeline from the
[stitching tutorial](stitching.md):

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:segmentation_fn"
```

## Step 2: check the partition layout

A chunk is one atomic write object, so tiles that share an output chunk must travel in
the same job: [effective parallelism follows the output's
chunking](../getting_started/6_iterators.md#distributed-runs), not the tiling. Derive
the output label with the defaults and the constraint shows itself:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:fat_partitions"
```

Fifty tiles, but two fat chunks: two working jobs, and two empty no-ops. This
`partition_indices` listing is the pre-flight check worth running before anything is
submitted. Chunk the output to match the work instead:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:aligned_partitions"
```

```python exec="true" html="1" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:partition_figure"
```

## Step 3: run the three-phase recipe

Schedulers like Fractal run distributed work as **init → parallel tasks →
consolidate**, and the iterator verbs map onto those phases one to one:
`prepare_jobs` performs any setup the run needs (wiping stale scratch state from
earlier runs first) and returns one JSON-ready argument set per non-empty partition;
each parallel task rebuilds the identical iterator and runs its own share; the
consolidate task's `finalize()` is the one global step.

<!-- Figure 12 — the three-phase distributed recipe -->
<div class="ngio-diagram">
<svg viewBox="0 0 640 278" style="display:block;width:100%;height:auto" role="img" aria-labelledby="f12t f12d">
<title id="f12t">The three-phase recipe for a distributed iterator run</title>
<desc id="f12d">An init task builds the iterator and calls prepare_jobs, which returns one set of arguments per partition. Each parallel task rebuilds the identical iterator and runs for_job(i).segment(func) over its own chunk columns, so no two tasks share a write unit. A final consolidate task calls finalize() once, which resolves the pyramid, stitches any banked labels and joins partial tables.</desc>

<g style="fill:var(--ngio-accent-soft);stroke:var(--ngio-accent)">
    <rect x="16.5" y="8.5" width="15" height="14" rx="3"></rect><rect x="216.5" y="8.5" width="15" height="14" rx="3"></rect><rect x="456.5" y="8.5" width="15" height="14" rx="3"></rect>
</g>
<g text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--ngio-accent-ink)"><text x="24" y="19">1</text><text x="224" y="19">2</text><text x="464" y="19">3</text></g>
<g style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.09em;fill:var(--md-default-fg-color--light)">
    <text x="38" y="19">INIT TASK</text><text x="238" y="19">ONE TASK PER PARTITION</text><text x="478" y="19">CONSOLIDATE</text>
</g>

<rect x="16.5" y="32.5" width="176" height="229" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
<rect x="30.5" y="64.5" width="148" height="58" rx="8" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5"></rect>
<g style="font-family:'JetBrains Mono',monospace;font-size:9.5px;fill:var(--md-default-fg-color)">
    <text x="40" y="82">SegmentationIterator(…)</text><text x="46" y="97">.by_chunks()</text><text x="46" y="112">.prepare_jobs(4)</text>
</g>
<rect x="32.75" y="147.75" width="145.5" height="88.5" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
<path d="M32 170h147M32 192h147M32 214h147" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></path>
<g style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--md-default-fg-color)">
    <text x="44" y="164">job_index=0</text><text x="44" y="186">job_index=1</text><text x="44" y="208">job_index=2</text><text x="44" y="230">job_index=3</text>
</g>

<g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M193 159H197V76H210M210 71l6 5-6 5"></path>
    <path d="M193 181H200V124H210M210 119l6 5-6 5"></path>
    <path d="M193 203H203V172H210M210 167l6 5-6 5"></path>
    <path d="M193 225H206V220H210M210 215l6 5-6 5"></path>
</g>

<rect x="216.5" y="32.5" width="216" height="229" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
<g style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--md-default-fg-color)">
    <text x="232" y="80">for_job(0).segment(func)</text><text x="232" y="128">for_job(1).segment(func)</text><text x="232" y="176">for_job(2).segment(func)</text><text x="232" y="224">for_job(3).segment(func)</text>
</g>
<g style="fill:var(--ngio-surface);stroke:var(--ngio-line-strong)">
    <rect x="384.5" y="62.5" width="44" height="26"></rect><rect x="384.5" y="110.5" width="44" height="26"></rect><rect x="384.5" y="158.5" width="44" height="26"></rect><rect x="384.5" y="206.5" width="44" height="26"></rect>
</g>
<g style="fill:var(--ngio-accent)" fill-opacity=".35">
    <rect x="384.5" y="62.5" width="11" height="26"></rect><rect x="395.5" y="110.5" width="11" height="26"></rect><rect x="406.5" y="158.5" width="11" height="26"></rect><rect x="417.5" y="206.5" width="11" height="26"></rect>
</g>
<g style="stroke:var(--ngio-line-strong)" stroke-width="1" opacity=".8">
    <path d="M395.5 62.5v26M406.5 62.5v26M417.5 62.5v26M384.5 75.5h44"></path>
    <path d="M395.5 110.5v26M406.5 110.5v26M417.5 110.5v26M384.5 123.5h44"></path>
    <path d="M395.5 158.5v26M406.5 158.5v26M417.5 158.5v26M384.5 171.5h44"></path>
    <path d="M395.5 206.5v26M406.5 206.5v26M417.5 206.5v26M384.5 219.5h44"></path>
</g>

<g fill="none" style="stroke:var(--ngio-line-strong)" stroke-width="1.5">
    <path d="M433 76h8v96M433 124h8M433 172h15M433 220h8v-48M448 167l6 5-6 5"></path>
</g>

<rect x="456.5" y="32.5" width="167" height="229" rx="10" style="fill:var(--ngio-sunk);stroke:var(--ngio-line)"></rect>
<rect x="472.5" y="64.5" width="135" height="36" rx="8" fill="none" style="stroke:var(--ngio-accent)" stroke-width="1.5"></rect>
<text x="540" y="88" text-anchor="middle" style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--md-default-fg-color)">finalize()</text>
<g style="fill:var(--ngio-blue)">
    <rect x="478" y="148" width="12" height="8" rx="1.5"></rect><rect x="478" y="158" width="22" height="9" rx="1.5"></rect><rect x="478" y="169" width="34" height="9" rx="1.5"></rect>
</g>
<rect x="548.75" y="148.75" width="38.5" height="29.5" rx="2" fill="none" style="stroke:var(--ngio-magenta)" stroke-width="1.5"></rect>
<path d="M548 158h40M561 149v30M574 149v30" style="stroke:var(--ngio-magenta)" stroke-width="1.2"></path>
<g style="font-family:'IBM Plex Sans',sans-serif;font-size:11.5px;fill:var(--md-default-fg-color--light)">
    <text x="472" y="200">resolves the pyramid,</text><text x="472" y="216">stitches banked labels,</text><text x="472" y="232">joins partial tables.</text>
</g>
</svg>
</div>

Every phase rebuilds the identical iterator from scratch — cheap, because
[construction is metadata-only](../getting_started/6_iterators.md#distributed-runs) —
and derives its share on its own:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:build_iterator"
```

This run stitches, so `prepare_jobs` is **required** — the stitch scratch has to be
created once, before any job writes, and the init step is that moment. (For a plain,
unstitched writer it is optional: `for_job` and `finalize` alone are enough.) It
returns the parallelization list, with empty partitions already dropped:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:init_task"
```

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:parallel_tasks"
```

A slice's `segment` deliberately does **not** finalize: until the gather runs, only
the iterated level is up to date, and the banked tiles are not yet reconciled. The
consolidate task runs the one global step — it verifies every expected bank exists (a
half-finished run errors, naming the tiles that never banked), resolves the seams,
and rebuilds the pyramid:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:consolidate_task"
```

The result is bit-identical to the serial `with_stitch()` run of the
[stitching tutorial](stitching.md). Three properties are worth knowing:

- **A failed job never destroys the others' banks** — re-run just that job (banking
  is idempotent) and gather as planned.
- **Every step validates a plan fingerprint** stamped at init: change the tiling,
  halo, stitch config, or `n_jobs` between phases and the run fails loudly.
- **Every job must use the same `n_jobs` and the same iterator construction** — the
  fingerprint catches most drift, but a custom seam matcher or the function itself
  cannot be fingerprinted, so declare the identical chain in every phase.

## Step 4: measure across jobs

The read-only iterators end in a *global join* that per-job runs
[cannot reproduce piecewise](../getting_started/6_iterators.md#distributed-runs), so
their topic verbs are partition-aware: on a `for_job` slice, `measure`
banks the job's raw pre-join records as a *partial* and returns `None`, and
`finalize()` runs the single global join and returns the table — the same three-phase
recipe, verb for verb:

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:measure_fn"
```

```python exec="true" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:measure_distributed"
```

### Sanity check: read the table back

```python exec="true" html="1" source="material-block" session="distributed"
--8<-- "docs/snippets/tutorials/distributed_processing.py:read_table_back"
```

There are slightly more rows than objects: the 2×2 blocks split some objects, and a
split object is measured by both sides. Every row is stamped with the `roi_index` /
`roi_name` it came from, and reconciling the duplicates is a declared join away — the
[feature extraction tutorial](feature_extraction.md) shows one.

## Next steps

- [Iterators guide](../getting_started/6_iterators.md#distributed-runs) —
  the partition model in full: why jobs need no locks, and what `finalize` refuses.
- [Stitching](stitching.md) — the serial version of this run.
- [Feature extraction](feature_extraction.md) — joins, halos and duplicate-row
  reconciliation for `measure`.
