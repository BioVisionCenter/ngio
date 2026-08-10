# Measuring ngio

Performance work here is split in two, because the two halves are different
kinds of thing and want opposite things from their data.

## `tests/performance/` — the gate (a test)

Asserts exact store-operation counts against committed baselines. Answers *"did
this change make ngio do more work?"* — pass/fail, no thresholds, runs in CI
with everything else.

It works as a gate because ngio's regressions are algorithmic — metadata
re-parsed per call, one group opened per well, a graph executed twice — so they
are exact integers with zero variance. Counts are also backend-independent, so
a local measurement predicts the cost on S3 where every op is a network
round-trip; that is checked rather than assumed, by running every scenario
against both a local and an in-memory store.

```bash
pixi run -e test11 pytest tests/performance
pixi run -e test11 pytest tests/performance -p no:xdist --update-baseline
```

Fixtures are tiny and **uncompressed** on purpose: exact byte counts matter more
than realism, and a numcodecs bump must not be able to move a baseline.

There is no history file — the baselines are committed JSON, so git already is
the history:

```bash
git log -p tests/performance/baselines/local.json
```

## `benchmarks/` — the suite (an experiment)

Answers *"how does this behave at scale, and will it fit in memory?"* Run it
when you are investigating something, not on a schedule.

```bash
pixi run -e dev python -m benchmarks
pixi run -e dev python -m benchmarks --list          # blocks and their axes
pixi run -e dev python -m benchmarks --only consolidate
pixi run -e dev python -m benchmarks --keep          # reuse generated data
pixi run -e dev python -m benchmarks --csv out.csv   # append results
```

No pixi environment of its own and no dependencies beyond ngio — `time` and
`tracemalloc` are stdlib. Nothing is committed and nothing gates; the numbers
depend on the machine, so compare within one run rather than across weeks.

### Axes

A **block** is one measured operation. It declares its **axes** as data, and
the runner measures every point of their cartesian product. That is the whole
design: axis values are not literals buried in a loop, so they can be swept
from the command line and land as real CSV columns.

```bash
pixi run -e dev python -m benchmarks --axis consolidate.z=16,64,256,1024
pixi run -e dev python -m benchmarks --axis mode=numpy      # every block with a `mode`
pixi run -e dev python -m benchmarks --only layout --axis layout.layout=sharded --axis z=16,64
pixi run -e dev python -m benchmarks --repeats 7
```

`--axis` **replaces** an axis rather than extending it. Unqualified, it applies
to every selected block that declares that name; qualified with `block.`, to
one. A name no selected block declares is an error — an override that silently
measures the defaults is worse than no override.

Axes come in two kinds, and the declaration form is the meaning:

| form | kind | the CLI can |
| --- | --- | --- |
| `"z": [16, 64, 256]` | **open** | subset, *and* name values that were never declared |
| `"layout": {"sharded": {...}}` | **closed** | subset only |

Open values are scalars, so `--axis z=1024` re-parses as an int and works.
Closed values are structures — a bundle of `create_empty_ome_zarr` kwargs — so
their labels are the vocabulary. That is deliberate: chunk and shard shapes are
not independent knobs (`sharded` + `uncompressed` is not a case anyone wants),
and a shard shape is not something to type at a shell prompt.

`--list` prints every block, its axes, their kind, and their values. With the
parameterization as data, it is the only place the sweepable surface is
discoverable without reading six files.

### The blocks

| block | axes | question |
| --- | --- | --- |
| `consolidate` | `mode` x `z` | which pyramid mode should I use, and will it fit? |
| `layout` | `layout` x `z` | same bytes, different chunk/shard shape |
| `roi` | `alignment` x `size` | chunk-aligned vs straddling reads |
| `algorithms` | `kernel` x `n` | scaling curves for ngio's own algorithms |

`algorithms` reports a series so the *shape* is visible — one timing cannot
tell O(n) from O(n²), four can. Its three kernels have different useful `n`
ranges, so cases outside a kernel's range raise `Skip` and the run reports how
many it dropped; a bounded sweep must never read as a complete one.

`get`/`set` are deliberately absent: they are a thin layer over zarr, adding a
roughly constant ~0.8 ms, so timing them mostly re-measures zarr.

Fixture data is seeded and generated through ngio's **public API only**, so the
blocks run unmodified inside an environment holding a different ngio. It is
also deliberately *compressible*: uniform data compresses ~2000:1 and pure noise
1:1, either of which would make the `layout` comparison meaningless without
looking wrong. The generator is tuned to ~1.8x, matching the ~1.7x of a real
sample image. Fixture names are derived from the spec, so two blocks asking for
the same image share one store under `--keep`.

`algorithms` is the one block that must reach into private modules — it
measures internal algorithms, which have no public entry point. Those imports
stay inside `run`, so a version where the paths differ degrades that block to
"unavailable" rather than failing the run. `blocks/__init__.py` holds module
*paths* rather than imported modules for the same reason: a block that cannot
import must not take discovery down with it.

### Peak memory is the point, not an extra

The three pyramid consolidation modes differ mainly in what they hold at once,
so a timing alone cannot tell you which one survives your data:

| mode | 8 MB | 32 MB | 128 MB |
|---|---|---|---|
| `dask` (default) | 4.7 MB | 6.5 MB | 11.6 MB |
| `numpy` | 12.7 MB | 48.1 MB | **192.3 MB** |
| `coarsen` | 12.0 MB | 12.8 MB | 20.5 MB |

`numpy` peaks at roughly 1.5x the data and scales with it — a hard ceiling. The
other two are chunk-bounded and stay flat. `numpy` is also the *fastest*, by
around 2.4x over `dask`, which is only a sensible trade if the level fits in
RAM.

Timing and allocation are measured in **two separate phases**. `tracemalloc`
hooks every allocation and inflates allocation-heavy code, and it does not
inflate every case equally — the mode that allocates most is penalised most,
which is exactly the comparison this block exists to make. Peak is the max over
its runs, not the last one, because the question peak answers is "will this
fit", and that is a worst case.

`tracemalloc` counts Python-side allocations only; it misses memory numpy and
dask acquire outside the Python allocator. Treat it as a strong relative signal
between cases, not an absolute footprint.

### Comparing environments

```bash
pixi run -e dev python -m benchmarks --only consolidate \
  --env current --env 'ngio==1.0.0' --env 'current,zarr==3.0.6'
```

One product; the outer factors are realized by re-exec, the inner ones
in-process. An ngio version cannot be varied inside a running interpreter, so
the parent spawns one child per `--env` and each child sweeps its axes
normally.

An `--env` spec is a comma-separated requirement list where `current` means
this working tree, so "did my optimization actually help" is answerable without
publishing anything. Each runs in its own isolated environment via `uv`, which
is therefore required for this flag. An environment that fails to install is
reported and skipped rather than aborting the rest, and its cases render as `—`.

Being able to pin *anything*, not just ngio, is the point: installing an older
ngio also resolves *its* dependency versions, so a difference between versions
can come from zarr rather than from ngio. Pinning both is the only way to tell.
The CSV records `zarr` and `python` on every row for the same reason — check
those columns before concluding anything about an ngio change.

`--keep` is never forwarded to a child. A fixture written by ngio 1.0.0 and read
back by the working tree is a different experiment from the one you asked for,
so every child gets its own temp root.

Neither is `--config`: it may itself set `envs`, so a child reading it would
recurse. The parent resolves the file and the command line together and lowers
the result back to plain `--only`/`--axis`/`--repeats`/`--warmup` flags, which
is the only thing a child is ever told.

### An experiment as a file

An invocation is an experiment, and one that lives only in shell history cannot
be committed next to the CSV it produced, diffed against last month's, or
pasted into an issue. `--config` is that file.

```bash
pixi run -e dev python -m benchmarks --config benchmarks/experiments/consolidate-ceiling.toml
```

```toml
blocks  = ["consolidate"]
csv     = "consolidate-ceiling.csv"   # relative to THIS file, not the CWD
envs    = ["current", "ngio==1.0.0"]
repeats = 7
keep    = true

[axes.consolidate]
mode = ["numpy", "dask"]
z = [16, 64, 256, 512, 1024]
```

`experiments/reference.toml` documents every key with its default, entirely
commented out — copy it and uncomment what you need. Keep plain settings above
the first `[axes]` table: TOML binds a bare key to whichever table precedes it,
so a `repeats` below `[axes.consolidate]` becomes an *axis* named `repeats`.

One `[axes.<block>]` table per block, whose keys are that block's axes. Every
entry names its block: the unqualified `--axis z=16` the command line accepts
has no config equivalent, on purpose. A file is read later by someone who was
not there, and "whichever blocks happen to have a `z`" describes the suite at
the moment it was written rather than the experiment. Two blocks needing the
same value get one entry each. Every key is optional and an unknown one is an
error.

Precedence, in one sentence: **the command line beats the file.**

```bash
--config ceiling.toml --axis z=2048   # one z, not five
--config ceiling.toml --no-keep       # temp dir, despite keep = true
```

`--keep` and `--quiet` have `--no-` forms so a `true` in a file can be switched
off. `--list` applies the config, so it is a dry run of the experiment rather
than a catalogue of the defaults — worth doing before a long sweep.

A config is **not** a second way to declare axes; it can only ever *replace*
one a block already declares. That is what keeps the file honest: an entry
lowers to exactly the same override a `--axis` flag produces and goes through
the same code, so the two cannot drift, and `AXIS_FIELDS` in `_output.py` needs
no config hook — a config cannot invent a column.

Results are gitignored (`benchmarks/experiments/*.csv`) while the toml is not.
The recipe is the committable artefact; the numbers are machine-dependent. For
the same reason there is no `config` column in the CSV and no provenance
sidecar — the toml sitting next to the CSV is the record, and `env`,
`ngio_version` and `zarr` already pin what it resolved to.

### CSV schema

```
env,ngio_version,python,zarr,platform,block,case,
alignment,kernel,layout,mode,n,size,z,seconds,peak_mb,note
```

One row per case, environment repeated. Redundant, but it makes a single file
answer cross-environment questions with no joins, and runs append so several
accumulate. `env` holds the *requested* spec and `ngio_version`/`zarr` the
*resolved* ones — requested ≠ resolved is exactly the trap above.

Axis columns hold **labels**, not values: `layout` is the string `sharded`,
never a repr of a kwargs dict. Labels are unique per axis and are already what
`--axis` speaks, so a CSV cell and a CLI token are the same token, and
`df.pivot(index="z", columns="mode", values="peak_mb")` is a one-liner.

The header is the union of axis names across every block, so it is
deterministic and appends line up. A file whose header does not match **raises**
rather than being appended to.

## Adding to either half

A scenario or block earns its place by informing a decision — *is this doing
more work than it should*, or *which option should I choose*. Ones that merely
produce a number do not. Keep both lists short.

A new block is one file in `benchmarks/blocks/`, one line in
`blocks/__init__.py`, and any new axis name added to `AXIS_FIELDS` in
`_output.py`. It declares `AXES`, an optional `REPEATS`, and:

```python
def run(root: Path, **values) -> Measured:
    """Set up, then return the callable to measure."""
```

Everything outside the returned callable is excluded from the measurement —
the same split as `Scenario(setup, run)` in `tests/performance/scenarios.py`,
written as one function because a block's setup and its measured call always
share state.
