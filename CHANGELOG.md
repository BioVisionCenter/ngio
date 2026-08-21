# Changelog

## [v1.1.0]

**Highlights**:

- **The IO layer got much faster**: metadata is no longer re-read and re-decoded on every access (~250x on the hottest property, zero store reads when cached), dask writes align to the array's write unit (no more whole-shard read-modify-writes, and `zarrs` finally engages on local stores), and plate walking is flat instead of quadratic.
- **Iterators run at scale**: `map`/`reduce` fan out on threads or processes (`ThreadedMapper`/`ProcessMapper`, lock-free via conflict-free waves), read context with `with_halo(...)` at unchanged parallelism, batch patches for neural networks (`BatchedMapper`, `iter_batched`), and distribute across SLURM/Fractal tasks with the `prepare_jobs → for_job → finalize` recipe.
- **New capabilities**: `stitch=True` merges objects split across region boundaries on *any* ROI list — grids, overlapping FOVs, ragged tables, tiled masked objects; the new `ObjectDetectionIterator` turns a tile-by-tile detector into one deduplicated ROI table ([#111](https://github.com/BioVisionCenter/ngio/issues/111)); `FeatureExtractorIterator.measure` joins per-region measurements into a single `FeatureTable`.

### Features

- `ThreadedMapper` and `ProcessMapper` parallelize `map`/`reduce`. Writes run in conflict-free waves, and wave order is the canonical write order for every mapper — serial included — so overlapping writes land identically everywhere. The default stays serial.
- `BatchedMapper(batch_size=...)` stacks patches into one `(B, ...)` array per `func` call, for neural-network inference; `iter_batched(batch_size)` is the manual batched loop.
- `reduce(func, mapper=...)` returns per-ROI results, writing nothing.
- `with_halo(x=, y=, ...)` reads a grown region and writes only the core; write footprints — and so parallelism — are unchanged, and misuse refuses loudly.
- `stitch=True` merges objects split across region boundaries into one id, on any ROI list — grids with a halo, overlapping FOVs, ragged tables — compacting ids to a dense `1..N`; an interrupted run leaves a valid, merely over-split label. On `MaskedSegmentationIterator` it merges sub-objects within one mask (no `UniqueLabelsTransform` needed; combining raises).
- Distributed runs: `prepare_jobs(n_jobs)` → `for_job(**args).map(func)` per SLURM/Fractal task → one gather (`finalize()` for writers, `merge_partials()` for the read-only iterators). Partitions never share a write unit, every step validates a plan fingerprint, and the result matches a serial run.
- `ObjectDetectionIterator.detect(func)` ([#111](https://github.com/BioVisionCenter/ngio/issues/111)) runs a `(patch) -> list[Roi]` detector per tile, anchors the boxes to world coordinates, deduplicates by NMS (`nms=NmsConfig(...)`), and returns one renumbered `RoiTable`; `Roi.anchor` is public.
- `FeatureExtractorIterator.measure` joins per-ROI `func(image, label, roi)` results into one `FeatureTable` (custom `coalesce` supported).
- `Label.relabel_sequential()` renumbers any label to a dense `1..N` in one pass.
- `merge=` on writes combines with what is on disk: `"max"`/`"min"`/`"sum"` (order-independent), `"keep_nonzero"`, a callable, or a policy. `MaskTransform` fills on read and `MaskMerge` protects on write, replacing the four masked pipe classes (`get_roi_masked`/`set_roi_masked` unchanged).
- `UniqueLabelsTransform` gives each region a disjoint id block — derived, not counted, so parallel-safe and idempotent.
- The transform contract is two methods, `on_get(array, ctx)`/`on_set(array, ctx)`, importable from `ngio.transforms`; a transform that silently materialises a dask array fails loudly.
- Canonical `iter()`/`map()`/`reduce()` (numpy backend; `*_as_numpy` stay as aliases) and tiling calls that say what they tile by: `by_grid(size_*, stride_*, tail=...)`, `by_blocks(num_*)`, `by_chunks()`, `by_write_units()`.
- `consolidate(mode="auto")` takes the 3–5x-faster in-memory path exactly where it is provably identical to the chunked one.
- `consolidate(regions=...)` rebuilds only the pyramid regions that changed, byte-identical to a full rebuild; writing iterators use it automatically, and `Image.track_writes()` records the regions for you.
- `NgioConfig` gained three sections, each with a public model: `zarr` (`ZarrConfig`), `dask` (`DaskConfig`), and `consolidation` (`ConsolidationConfig`).
- Every `max_workers` accepts `"auto"`; `refresh()` on containers and plates re-reads all held metadata; the scheduling primitives (`plan_waves`, `canonical_unit_order`, `TailPolicy`, …) are public in `ngio.iterators`.
- New public names: `NgioFutureWarning`, `ChannelSlicingInputType` (`ngio.images`), `ConsolidationMode` and `RegionsLike` (`ngio.common`), and `AbstractImage.write_granularity()` — the shape zarr writes atomically.

### Fixes

- Metadata updates no longer drop what they were not about ([#231](https://github.com/BioVisionCenter/ngio/issues/231)): `set_axes_units`/`set_axes_names` silently reset OMERO channels (NGFF 0.5) and a label's source metadata; subclass state is now preserved.
- ngio's own writes could be invisible on a store carrying consolidated metadata — ngio now ignores `.zmetadata` everywhere (it never wrote any).
- A dask write could lose updates when the write unit exceeded dask's block budget; the budget now covers one write unit, and the process-global `DASK_STORE_LOCK` (which never worked across processes) is gone — `da.to_zarr` gives each write unit exactly one writer.
- Writing through a chain of two or more transforms now inverts the chain — it applied the inverses in forward order, silently wrong for any user chain on a write path.
- A merge can no longer broadcast a wrong-shaped patch or hand zarr a promoted dtype (a float patch under `merge="max"` used to wrap-cast, `-1.0` becoming `65535`); integer `"sum"` still wraps within the dtype — numpy addition, documented.
- The zoom transform's edge cases are pinned: negative starts clamp like the read, empty selections zoom to empty patches, and a scaled list selection raises instead of silently stretching non-adjacent pixels.
- `cache=False` re-probes: a container's remembered "no tables" answer and a table's memoised type could outlive writes made through another handle; both now honour the cache flag.
- `normalize_anndata` dropped `.raw` while normalising `obs`; it is carried through again (the result still shares its other components with the input rather than copying them).
- A failed `map` on a stitched iterator cleaned up the scratch even when it had only *opened* a prepared root (a resumed run, or the gather step) — one failure could delete every job's banked predictions. Cleanup now runs only when the failing run created the scratch itself.
- The write-conflict graph keyed output arrays by handle identity, so two `zarr.Array` handles onto one stored array were never compared — a latent lost update under parallel mappers. It now keys on the stored array (store + path).
- A whole-array dask write now verifies its block grid gives each write unit exactly one writer, raising loudly instead of leaning on a silenced dask warning.
- `relabel_sequential` and the stitch compaction walk sharded labels per shard, not per inner chunk (each inner-chunk write was a full-shard read-modify-write).
- A `BatchedMapper` reduction on a ragged batch raises: the result is computed on padded patches, so the padding had already leaked into the values.
- Invalid sharding/compression setups fail before anything is written, with named errors instead of a raw zarr error after the group metadata already landed (a half-built container): shards on OME-Zarr 0.4, an explicit shard shape with `chunks="auto"`, and a cross-format derive inheriting codecs or shards (`compressors="auto"` picks the target format's default; `shards="auto"` on a 0.4 target means no sharding).
- `find_dimension_separator` accepts the v2 chunk-key encoding on zarr v3 arrays, so deriving from such stores no longer fails.
- Serialising a ROI table warns once per unknown extra column instead of once per ROI, the message explains the round-trip consequence, and `confidence` — written by `detect` — is recognised and no longer warns.
- Stale table names are skipped by typed listings instead of *creating* a group for the missing table; shards clip to whole chunk multiples instead of producing a geometry zarr rejects; `mode="coarsen"` asked to upsample raises a named error instead of a bare divide-by-zero; a dropped channel selection is no longer validated before the removal that exempted it; a zero-sized axis reports itself instead of blaming the tail policy; stitch plan warnings fire at the start of `map`, pointing at the caller, not mid-run from a worker.

### Behaviour changes

- The `dask[array]` floor rises to 2025.12 — the first release whose `to_zarr` both aligns writes to the target's write unit and warns (rather than raises, [dask#12159](https://github.com/dask/dask/issues/12159)) on the rechunk it does so.
- **The promotion out of `experimental` renames the iterator surface** — one-time renames, no shims (the 1.0 surface was explicitly experimental): `grid()` → `by_grid()`, `by_chunks(grid="write")` → `by_write_units()`, `overlap_xy` → `overlap_x`/`overlap_y`, `check_if_chunks_overlap` → `check_if_write_units_overlap` and `require_no_chunks_overlap` → `require_no_write_units_overlap` (both now measured on the write target, at shard granularity when sharded), `post_consolidate()` → `finalize()`, `ImageProcessingIterator(input_channel_selection=)` → `channel_selection=`. `MapperProtocol` changed shape the same way (`(func, units)`; no known custom mappers exist).
- Serial `map` over overlapping write footprints runs in wave order, not ROI order — "who wrote last" is identical under every mapper; disjoint tilings are unaffected.
- A crashed stitched/distributed run can leave transient `_ngio_stitch`/`_ngio_partials` groups beside the resolution levels — unregistered, ignored by older ngio, wiped by the next run.
- `cache=True` actually caches: metadata is held for the object's lifetime (outside writes need `refresh()`), `image.dimensions` is fixed per object, and caching composes with the atomic plate/well operations. `cache=False` is unchanged and remains the default.
- `max_workers=0` raises instead of running silently serial; `by_grid(tail="drop")` raises when it would drop every tile; a writing `map` no longer holds every written patch in memory until it returns.
- ngio no longer wraps a plain local store in `NgioStore` when the retry policy is a no-op (the wrapper cost `zarrs` for nothing), and a codec pipeline that silently fell back now warns once per store type.
- `measure` returns an empty `FeatureTable` when zero objects are found — the same contract as `detect` — instead of raising.
- `stitch(compact=True)` warns when the final renumbering reaches ids outside the iterated ROIs: the compaction walks the whole label, so external references keyed by those ids (feature tables, masking ROI tables) go stale.

### Deprecated

All removals scheduled for `ngio=1.2`. The default-flip entries warn as `NgioFutureWarning` (Python hides `DeprecationWarning` from end users); the rest as `NgioDeprecationWarning`.

| Deprecated | Use instead |
|---|---|
| `set_axes_units` ([#232](https://github.com/BioVisionCenter/ngio/issues/232)) | `set_space_unit` / `set_time_unit` (the batch form silently resets the unit you omit) |
| `ngio.experimental.iterators` | `ngio.iterators` (one more release of forwarding for the Fractal tasks ecosystem) |
| `iter_as_dask`, `map_as_dask`, `reduce_as_dask`, `iter(data_mode="dask")` | the numpy verbs; `Image.get_as_dask` for lazy whole-region access |
| bare `iter()` defaulting to dask | pass `data_mode="numpy"` now; numpy becomes the default |
| legacy transform methods (`get_as_numpy_transform` & co.) | the two-method `on_get`/`on_set` contract |
| the ROI and masked pipe classes (8 names) | bare pipes with `roi=`, `MaskTransform` in `transforms=`, `merge=MaskMerge(...)` |
| `consolidate()` without a `mode` | becomes `mode="auto"`; pass `mode="dask"` to pin today's behaviour |
| plate fan-outs without `max_workers` | becomes `"auto"`; pass `max_workers=1` to pin serial reads |

### Removed

- **The surfaces v1.0.0 deprecated for `ngio=1.1` are gone** (except the `experimental.iterators` alias — see Deprecated): the `conctatenate_tables` typo alias, `set_axes_unit`, the `levels_paths=`/`validate_paths=` keyword aliases, and every `*_async` variant — `plate.get_images_async()` becomes `plate.get_images(max_workers="auto")`.

### Performance

- Metadata is no longer re-read and re-decoded per access: `image.dimensions` — read once per ROI by every iterator — drops from ~520µs to ~2µs, and to zero store reads under `cache=True`.
- Plate walking is flat instead of quadratic: `create_empty_plate` 733 → 178 metadata reads (14.4 MB → 128 KB written on a 384-image plate); per-well image paths 75 → 9 reads.
- Dask writes align to the write unit: sharded writes stop read-modify-writing whole shards (128 chunk reads → 2 on a 1 MB write); in-memory blocks are capped at 8 MiB for a ~75% peak-memory cut at no wall-clock cost.
- `zarrs` actually engages on local stores: `set_array` 76 → 17 ms, `get_as_numpy` 39 → 9 ms on a 32 MB image.
- Consolidation reads less: sharded sources read at their write unit (324 chunk reads → 8), the numpy path chains levels through memory, coarsening drops its float64 intermediate (~3× peak), and the dask mode stops computing every source chunk twice.
- Tables and overlap checks: type-filtered listings are one memoised pass under `cache=True` (95 reads → 2 warm; `cache=False` re-reads per call, by contract), ROI tables rebuild their DataFrame only on change, and `check_if_regions_overlap` sweeps instead of scanning all pairs (2,048 ROIs: 1,083 ms → 31 ms; `check_if_write_units_overlap` remains all-pairs).

### Chores

- The exact operation-count baselines behind the numbers above are committed under `tests/performance/` and gated in CI.
- Fixed the Windows CI test failures from the `v1.1.0a1` build: three tests asserted the bare-`LocalStore` bypass unconditionally, but on Windows `NgioStore` always wraps local stores (its sharing-violation retry is unconditional). The tests are now platform-aware, and the Windows-wrap decision is covered on every platform via a monkeypatched test.

## [v1.0.1]

Concurrency and Windows fixes. No API change — every `v1.0.0` call keeps working.

### Behaviour changes

- Lock files moved out of the Zarr store into a `<store>.ngio-locks/` directory beside it. **Upgrade every writer to a plate at the same time:** a `≤1.0.0` writer takes the old in-store paths, so mixed versions never exclude each other and one concurrent update is lost silently. Groups differing only after a dot (`foo.bar`, `foo.baz`) no longer share a lock. Locks are keyed on the store root, so a well opened directly with `open_ome_zarr_well` and the same well reached through its plate no longer share one — take atomic well operations through the plate. Old `.lock` files left inside a store by an earlier version are not cleaned up.
- `atomic_add_image` / `atomic_remove_image` warn on Windows that their lock is best-effort: `filelock` can hand it to two writers at once, so concurrent ones can lose an update — `v1.0.0` lost it silently. A single writer is unaffected. The warning is an error under `-W error` or `filterwarnings = ["error"]`.

### Fixes

- Concurrent writers no longer race on creating a group: `get_group(create_mode=True)` is a get-or-create, and `atomic_add_image` creates the well group under the plate lock. Two workers adding to the same well could fail with `NgioFileExistsError`.
- Windows: concurrent *reads* of a metadata file no longer fail with `PermissionError`. `v1.0.0` only retried the conflict when the error carried a Win32 code, which `os.replace` sets but `open()` does not.
- `Label.consolidate(mode="coarsen")` averaged label IDs instead of taking the maximum — the mean of labels 3 and 7 is 5, a label that never existed — and truncated on integer dtypes. `on_disk_zoom` did not forward `order` to the coarsening path.
- zarr 3.3 support. zarr 3.3 added a coalescing `get_ranges` and a synchronous store surface (`get_sync`, `set_sync`, `delete_sync`); ngio's store inherited all of them from `WrapperStore` forwarded straight to the wrapped store, so they ran with **no retry policy and no Windows sharing-violation retry** — unlike every other IO operation. They are now routed through the same retry path. Nothing reached these methods with zarr's defaults, so no released version could lose data over it; sharded arrays and the opt-in `FusedCodecPipeline` do.

### Chores

- Added a performance gate at `tests/performance/`: exact store-operation counts asserted against committed baselines, running in CI like any other test. See `tests/performance/README.md`.
- The performance gate counts zarr 3.3's new store surface, and its instrumentation check now covers `WrapperStore` as well as `Store` — the sync methods arrived on the former and a `Store`-only check could not see them. It also fails when zarr *removes* a hooked method, which would previously have zeroed a counter silently. Op counts are unchanged on zarr 3.1.6, 3.2.1 and 3.3.0.
- The op-count assertion is skipped when the counts differ *and* zarr is not the version the baselines were generated on, so an upstream zarr release no longer fails `CI (pip)`, which installs dependencies unpinned. The `test11` environment still asserts strictly on every PR.
- Linting moves from `pre-commit` to [`prek`](https://github.com/j178/prek), a drop-in reimplementation. `pixi run -e dev lint` is still the entry point; `pre-commit autoupdate` becomes `prek auto-update`.
- CI no longer depends on any Node 20 action.

## [v1.0.0]

First stable release. Everything deprecated in `v0.5.0` (each warned "will be removed in `ngio=0.6`") is now removed — that release became `1.0.0`.

### Removed

| Removed | Use instead |
| --- | --- |
| `OmeZarrContainer.image_meta` | `.meta` |
| `.levels_paths` | `.level_paths` |
| `.set_channel_percentiles(a, b)` | `.set_channel_windows_with_percentiles(percentiles=(a, b))` |
| `version=` on the plate/well create and derive functions | `ngff_version=` |
| `check_type=` on `get_table` | `get_table_as(name, TableCls)` or `get_*_table(name)` |
| `pixel_size=`, `xy_pixelsize=` on create/derive | `pixelsize=` (plus `z_spacing=`, `time_spacing=`) |
| `xy_scaling_factor=`, `z_scaling_factor=` | `scaling_factors=` |
| `labels=`, `channel_labels=`, `channel_wavelengths=`, `channel_colors=`, `channel_active=` | `channels_meta=` |
| `wavelength_id=`, `start=`, `end=`, `percentiles=`, `colors=`, `active=` on `set_channel_meta` | `channel_meta=ChannelsMeta(...)` |

`pixelsize` is now **required** on `create_empty_ome_zarr` and `create_ome_zarr_from_array`. The `pixel_size=` argument on the *getters* is unaffected: `pixel_size` selects a pyramid level, `pixelsize` is a value written on create.

### Behaviour changes

Same call, different result:

- `derive_image` inherits `dtype`, `dimension_separator` and `compressors` from the reference image instead of forcing `uint16` — deriving from a `float32` image no longer silently downcasts it.
- `add_table` and `write_table` keep the source table's backend instead of rewriting it as `anndata_v1` ([#207](https://github.com/BioVisionCenter/ngio/issues/207)). Pass `backend=` to convert.
- Opening a container no longer reads every pyramid level (`validate_arrays=False` by default), so a bad array fails on first access rather than at open.
- `open_image` and `open_label` default to `strict=False`, matching every other getter.
- `list_roi_tables` returns `[]` instead of raising when there are no tables.
- `get_masked_label(path=...)` resolves the masking label at the label's own pixel size, matching `get_masked_image`.
- `PixelSize`s with different `time_unit`s now compare unequal, and `==` against a non-`PixelSize` returns `NotImplemented`.

### Deprecated, removal in `ngio=1.1`

`conctatenate_tables` → `concatenate_tables`, `set_axes_unit` → `set_axes_units`, `levels_paths=` → `level_paths=`, `validate_paths=` → `validate_arrays=`, `ngio.experimental.iterators` → `ngio.iterators`, and every `*_async` plate/table function → the sync form with `max_workers=`.

### Migration (v0.5 → v1.0)

```python
create_empty_plate(store, ngff_version="0.4")                  # was version=
create_empty_ome_zarr(store, pixelsize=0.5,                    # was xy_pixelsize=
                      scaling_factors=(1.0, 2.0, 2.0))         # was *_scaling_factor=
ome_zarr.get_table_as("roi", RoiTable)                         # was check_type="roi_table"
ome_zarr.derive_image(store, channels_meta=["DAPI"], pixelsize=(ps.y, ps.x))
ome_zarr.set_channel_meta(channel_meta=ChannelsMeta.default_init(labels=["DAPI"]))
from ngio import SegmentationIterator                          # was ngio.experimental
plate.get_images(max_workers=8)                                # was get_images_async()
```

### Features

- The iterators are stable API: `from ngio import SegmentationIterator`.
- Configurable IO retries: `NgioConfig.io_retry` plus the `ngio.utils.retry_io` decorator. ngio's own `NgioError`s are never retried. See the Configuration page.
- `ngio.utils.NgioStore` wraps every zarr store ngio opens and applies that retry policy to all IO. `ZipStore` is now supported.
- `max_workers=` on the sync plate and table APIs replaces the separate async surface; `None` keeps the serial behaviour.
- A larger public namespace, including `MaskedImage`, `MaskedLabel`, `Channel`, `S3FSConfig`, `derive_ome_zarr_plate`, `__version__`, the `get_ngio_*_meta` readers and every error class. `AbstractBaseTable`, `ImplementedTables` and `write_table` are exported from `ngio.tables`, so a custom table type can be registered without private imports.
- `NgioTableValidationError` now subclasses `NgioValidationError`, so `except ValueError` catches it like its siblings; new `NgioKeyError`.

### Fixes

- Dask writes could silently drop data. `da.store(..., lock=False)` let two blocks read-modify-write the same chunk — or shard, when the target is sharded — concurrently, losing one update. This hit every region write that was not chunk-aligned and every sharded target, including pyramid consolidation. All `da.store` calls now share a lock; block compute stays parallel.
- Windows: concurrent access to a store no longer fails with `PermissionError: [WinError 5]`/`[WinError 32]`. A concurrent reader of `zarr.json` could break a writer's atomic rename; store operations now absorb these transient conflicts with a short bounded retry. No behaviour change on Linux or macOS.
- `import ngio` no longer raises `AttributeError` when an s3fs older than 2026.2.0 is installed.
- `concatenate_image_tables` built a wrong index: unnamed, and duplicated under `mode="lazy"`.
- `Roi.union`/`intersection` dropped ROI name `""` and label `0`; `Roi.from_values` now validates its inputs.
- Plate and well metadata `add_*`/`remove_*` mutated the receiver instead of returning a copy.
- `AxesSetup.from_ordered_list` silently dropped a non-canonical axis in some orders.
- Grid iterator ROIs now get unique names, and `by_chunks` with overlap ≥ chunk size raises `NgioValueError`.

### Packaging

- Ship `src/ngio/py.typed`. The PEP 561 marker was missing, so downstream type checkers ignored ngio's annotations.
- Real lower bounds on every dependency, exercised by a `test-min-deps` CI leg: `zarr>=3.1.6`, `numpy>=2.0`, `fsspec>=2025.3`, `anndata>=0.12.5`, `ome-zarr-models>=1.4` and the rest. `pandas` 3.x and `anndata` 0.13 are now allowed, the `requires-python` upper cap is gone, and unused `requests`/`distributed` are dropped.
- New `s3` extra: `pip install ngio[s3]`.

### Docs

- Rebuilt on [Zensical](https://zensical.org) with every code block executed at build time, plus new landing, glossary and Configuration pages.

## [v0.5.14]

### Fix
- Fix saving empty tables ([#99](https://github.com/BioVisionCenter/ngio/issues/99)). Empty tables now round-trip through both backends instead of raising a cryptic pandas error: (1) an empty ROI/masking table (zero ROIs) keeps its schema columns; (2) `_validate_cast_index_dtype_df` casts an empty index to the requested `str`/`int` type instead of rejecting it; (3) an empty ROI table with no backend materializes as an empty table rather than raising; and (4) `convert_pandas_to_anndata` no longer drops the numeric columns of a zero-row table (it now checks for zero columns rather than `DataFrame.empty`, which is also true for a zero-row frame).
- Fix `write_table` writing an empty table (or raising `FileNotFoundError`) when given a table returned by `open_table` whose data had not yet been loaded. `write_table` now materializes the table data before swapping to the destination backend, mirroring `TablesContainer.add`, so a table opened from one store can be copied to another via `write_table` with its data intact.
- Fix tables with missing (`None`/`NaN`) values in string columns failing to serialize to the AnnData backend. The `_check_for_mixed_types` and `_check_for_supported_types` guards now ignore missing values and classify a column from its non-null contents, so a string column containing `None` (or an all-missing column) is accepted — matching AnnData's native handling, which stores such columns as a categorical with `NaN` for the missing entries. This surfaced when copying a condition table (e.g. `get_table` + `add_table`), which re-serializes through the default AnnData backend. Note: the round trip normalizes missing-containing string columns from `object`/`None` to `category`/`NaN`.

## [v0.5.13]

### Feature
- Add a global `NgioConfig` / `get_config()` configuration system, loaded from `~/.ngio_config.json` by default or a path set via the `NGIO_CONFIG_PATH` env var (`.json` file). Both are exported from the top-level `ngio` package.
- Add configurable s3fs retry handling: `NgioConfig.s3fs.custom_retry_markers` lists error substrings that trigger a retry via a custom `s3fs.set_custom_error_handler`, applied through the new `ngio.utils.refresh_s3fs_config()`. The motivating use case is AWS clock-skew errors, but any error substring can be configured.

### Tests
- Migrate the S3 store test harness from a `moto[server]` subprocess to [`aiomoto`](https://github.com/owenlamont/aiomoto) in server mode (`aiomoto[pandas]` in the `test` extra). This also fixes a CI import crash on Python 3.13/3.14: `aiomoto` caps `aiobotocore`/`moto` and floors `s3fs`, so the universal (multi-platform) solve no longer backtracks `s3fs` to the ancient `0.4.2` (which lacks `set_custom_error_handler` and crashed `import ngio` at module load). CSV and Parquet table backends now round-trip on the S3 store under the mock.

### Fix
- Fix `derive_label` (and `OmeZarrContainer.derive_label`) rejecting an explicit `shape` that omits the channel axis. When the reference image has a `c` axis and `channels_policy` removes or overrides it (`"squeeze"`, `"singleton"`, or an integer), the up-front shape-length check failed before the channel policy was applied. The provided shape is now normalized to the reference dimensionality before pyramid computation, so a channel-less shape (e.g. `(z, y, x)` for a `(c, z, y, x)` image) is accepted. `channels_policy="same"` still requires the full shape.
- Fix `is_group_listable` wrongly reporting `True` for stores that cannot actually be listed (e.g. HTTP hosts without a directory index): zarr >= 3.1.6 swallows the listing error on `FsspecStore` and yields an empty listing instead of raising. The check now verifies that the group's own metadata document (`zarr.json` / `.zgroup`) — which must exist for any group that was successfully opened — appears in the store listing, distinguishing a broken listing from a genuinely empty group on any store type.
- `copy_group` now raises an error when the source listing does not contain the group's metadata document, instead of silently producing an empty copy from a non-listable store.

### Chores
- Harden GitHub Actions and scan workflows through `zizmor`.
- Rename the `pre-commit` pixi dev task to `lint`: the old name shadowed the `pre-commit` binary in `pixi run`, and its trailing `git add -u` silently staged the working tree and masked hook failures in the task's exit code.

## [v0.5.12]

### Fix
- Fix loading v0.4 HCS plates where the `version` key is absent from the plate-level metadata: the v0.4. V0.4 decoder now explicitly inject the version into the plate dict before constructing `PlateWithVersion`, so missing or `None` version values no longer cause a validation error.

### Refactor
- Remove redundant `version` field from `NgioPlateMeta`: the field is now a `@computed_field` property that delegates to `self.plate.version`, eliminating the need to keep two copies of the NGFF version in sync. The public `.version` attribute and `model_dump()` output are unchanged.

## [v0.5.11]

### Fix
- Remove eager uniqueness check on `wavelength_id` in `ChannelsMeta.default_init`: duplicate `wavelength_id` values are now allowed at creation time. `get_channel_idx` raises a clear error if a lookup by an ambiguous `wavelength_id` is attempted, directing users to select by label instead.

## [v0.5.10]

### Fix
- Replace `da.to_zarr` with `da.store(..., lock=False)` in pyramid writes (`_on_disk_dask_zoom`, `_on_disk_coarsen`) and region slice writes (`_ops_slices`). Dask >=2025.11's `to_zarr` re-derives chunks via `normalize_chunks(chunks="auto", ...)` and emits a `PerformanceWarning` (treated as error by ngio's filterwarnings) when the result is not a multiple of the target's chunks; `da.store` writes blocks 1:1.
- Copy object/string-dtype zarr arrays directly when consolidating groups: dask >=2025.11 raises `NotImplementedError` from auto-chunking for these dtypes, so they bypass dask and are copied via numpy.
- Set `auto_shard_zarr_v3` together with `zarr_write_format` on `anndata`'s global settings via a new `_update_anndata_global_settings` helper, so reading/writing tables works correctly when mixing zarr v2 and v3 in the same session on anndata 0.12.

### Chores
- Pin `anndata` to `>=0.12.0,<0.13.0`.
- Unpin `dask` (remove the `<2025.11.0` upper bound introduced in v0.4.5).

## [v0.5.9]

### Fix
- Fix AnnData reading over HTTP when directory listing is disabled: skip optional Zarr groups (`uns`, `obsm`, `varm`, etc.) that cannot be discovered without listing.
- Fix `ngff_version` not being propagated when deriving a plate: `derive_plate()` and `derive_ome_zarr_plate()` now default `ngff_version` to `None` and inherit the source plate's version when no version is explicitly provided.

## [v0.5.8]

### Fix
- Change tolerance when converting Roi to pixel coordinates to avoid machine precision dependent rounding issues.

### Tests
- Improve testing for ZoomTransform.
- Remove broad warnings filter for all tests.

### Chores
- Replace custom logger warnings with standard Python warnings for better integration with user applications.

## [v0.5.7]

### Fix
- Add docstrings to `ChannelSelectionModel` to allow for correct json schema generation.

## [v0.5.6]

### Fix
- Fix translation check in `_ngio_to_v04_multiscale` and `_ngio_to_v05_multiscale`: translations were incorrectly dropped when all values were negative or when positive and negative values cancelled out.
- Fix shape compatibility check in `_check_compatibility_of_shapes`: integer indices in the slicing tuple now correctly reduce the expected shape rank instead of inserting a spurious size-1 dimension.

## [v0.5.5]

### Features
- `Roi` now supports dict-like slice access: `roi["x"]` returns the slice for axis `"x"` and raises `KeyError` if the axis is not present.
- `Roi.get(axis_name, default=None)` now accepts an explicit `default` value, following the `dict.get` convention.
- New `Roi.update_slice(name, new_slice)` method: replaces the slice for an existing axis or appends a new one. Returns a new `Roi` instance.
- New `Roi.remove_slice(name)` method: removes the slice for a named axis. Returns a new `Roi` instance. Raises `NgioValueError` if the axis is not present.

### Chores
- Pin `mkdocs` to version <2.0 to avoid build errors in CI due to breaking changes in mkdocs v2, and incompatibility with material design theme.

## [0.5.4]

### Fix
- Remove file locking remove in `ZarrGroupHandler`, which was not used anywhere and is unnecessary in new lockfile release.
- Correctly set Zarr array dtype to array dtype in `create_ome_zarr_from_array`

## [0.5.3]

### Fix
- Fix bug in AnnData backend where "raw" entry with encoding-type "null" is written by default in newer anndata versions, which causes compatibility issues with older anndata versions. Now the "raw" entry is removed after writing if it has encoding-type "null".

## [0.5.2]

### Fix
- Fix critical bug in masking roi image handling causing incorrect results when image and mask have different pixel sizes.
- Fix bug in loading masking roi images when paths other than default are used.

## [0.5.1]

### Fix
- Fix bug causing incorrect channel metadata when creating an image.
- Fix correctly setting the space and time units when creating an image.
- Fix minor bug in `set_channel_windows_with_percentiles` method.

### Chores
- Improve logging consistency across the codebase.

## [v0.5.0]

### Features
- Add support for OME-NGFF v0.5
- Move to zarr-python v3
- API to delete labels and tables from OME-Zarr containers and HCS plates.
- Allow to explicitly set axes order when building masking roi tables.
- New metadata modification APIs for `Image`, `Label`, and `OmeZarrContainer`:
  - `set_channel_labels` - Update channel labels
  - `set_channel_colors` - Update channel colors
  - `set_channel_windows` - Update channel display windows (start/end values)
  - `set_channel_windows_with_percentiles` - Update display windows based on data percentiles
  - `set_axes_names` - Rename axes in the metadata
  - `set_axes_unit` - Set space and time units for axes
  - `set_name` - Set the image/label name in metadata
- Add translation support in all image/label creation and derivation APIs.

### API Breaking Changes

- New `Roi` models, now supporting arbitrary axes.
- The `compressor` argument has been renamed to `compressors` in all relevant functions and methods to reflect the support for multiple compressors in zarr v3.
- The `version` argument has been renamed to `ngff_version` in all relevant functions and methods to specify the OME-NGFF version.
- Remove the `parallel_safe` argument from all zarr related functions and methods. The locking mechanism is now handled internally and only depends on the
`cache`.
- Remove the unused `parent` argument from `ZarrGroupHandler`.
- Internal changes to `ZarrGroupHandler` to support cleanup unused apis.
- Remove `ngio_logger` in favor of standard warnings module.

### Migration Guide (v0.4 → v0.5)

#### Roi API Changes

The `Roi` class now uses a flexible slice-based model supporting arbitrary axes:

```python
# Old (v0.4)
roi = Roi(x=34.1, y=10, x_length=321.6, y_length=330)

# New (v0.5)
roi = Roi.from_values(slices={"x": (34.1, 321.6), "y": (10, 330)}, name=None)

# Accessing coordinates
# Old: roi.x, roi.y, roi.x_length, roi.y_length
# New: roi.get("x").start, roi.get("y").start, roi.get("x").length, roi.get("y").length
```

#### Argument Renames

```python
# compressor → compressors
# Old (v0.4)
create_empty_ome_zarr(..., compressor=Blosc())

# New (v0.5)
create_empty_ome_zarr(..., compressors=Blosc())

# version → ngff_version
# Old (v0.4)
create_empty_ome_zarr(..., version="0.4")

# New (v0.5)
create_empty_ome_zarr(..., ngff_version="0.4")
```

#### Removed Arguments

- `parallel_safe`: No longer needed, locking is handled internally
- `ngio_logger`: Use Python's standard `warnings` module instead

### Deprecations
- Standardized all deprecation warnings to indicate removal in `ngio=0.6`.
- Deprecated `set_channel_percentiles` method, use `set_channel_windows_with_percentiles` instead.

### Fix
- Fix bug in `consolidate` function when using coarsening mode with non power-of-two shapes.
- Fix HCS plate column name formatting to use standardized zero-padding (e.g., column `3` is now stored as `"03"`).
- Fix `_stringify_column` not passing `num_digits` parameter to `_format_int_column`.

### Documentation
- Fix incorrect and incomplete docstrings across the codebase:
  - `compute_masking_roi`: Added Args/Returns, fixed description (supports 2D, 3D, 4D).
  - `lazy_compute_slices`: Added Args/Returns sections.
  - `LabelsContainer.list`: Fixed description (was "Create the /labels group").
  - `build_masking_roi_table`: Added Args/Returns sections.
  - `TablesContainer`: Fixed class and method descriptions (were referencing labels instead of tables).
  - `NgioPlateMeta.add_well`: Fixed description (was "Add an image to the well").
  - `NgioPlateMeta.derive`: Fixed type annotation in docstring (`NgffVersion` → `NgffVersions`).
  - Added missing docstrings to several HCS helper functions.

## [v0.4.7]

### Fix
- Fix bug adding time axis to masking roi tables.
- Fix channel selection from `wavelength_id`
- Fix table opening mode to stop writing groups when opening in append mode.

## [v0.4.5]

### Fix
- Pin Dask to version <2025.11 to avoid errors when writing zarr pyramids with dask (see https://github.com/dask/dask/issues/12159#issuecomment-3548421833)

## [v0.4.4]

### Fix

- Fix bug in channel visualization when using hex colors with leading '#'.
- Remove strict range check in channel window.

## [v0.4.3]

### Fix

- Fix bug in deriving labels and image from OME-Zarr with non standard path names.
- Add missing pillow dependency.
- Update pixi workspace config.

## [v0.4.2]

### API Changes

- Make roi.to_slicing_dict(pixel_size) always require pixel_size argument for consistency with other roi methods.
- Make PixelSize object a Pydantic model to allow for serialization.

### Fix

- Improve robustness when rounding Rois to pixel coordinates.

## [v0.4.1]

### Fix
- Fix bug in zoom transform when input axes contain unknown axes (e.g. virtual axes). Now unknown axes are treated as virtual axes and set to 1 in the target shape.

## [v0.4.0]

### Features

- Add Iterators for image processing pipelines
- Add support for time in rois and roi-tables
- Building masking roi tables expanded to time series data
- Add zoom transformation
- Add support for rescaling on-the-fly masks for masked images
- Big refactor of the io pipeline to support iterators and lazy loading
- Add support for customize dimension separators and compression codecs
- Simplify AxesHandler and Dataset Classes

### API Changes

- The image-like `get_*` api have been slightly changed. Now if a single int is passed as slice_kwargs, it is interpreted as a single index. So the dimension is automatically squeezed.
- Remove the `get_*_delayed` methods, now data cam only be loaded as numpy or dask array.Use the `get_as_dask` method instead, which returns a dask array that can be used with dask delayed.
- A new model for channel selection is available. Now channels can be selected by name, index or with `ChannelSelectionModel` object.
- Change `table_name` keyword argument to `name` for consistency in all table concatenation functions, e.g. `concatenate_image_tables`,  `concatenate_image_tables_as`, etc.
- Change to `Dimension` class. `get_shape` and `get_canonical_shape` have been removed, `get` uses new keyword arguments `default` instead of `strict`.
- Image like objects now have a more clean API to load data. Instead of `get_array` and `set_array`, they now use `get_as_numpy`, and `get_as_dask` for delayed arrays.
- Also for `get_roi` now specific methods are available. For ROI objects, the `get_roi_as_numpy`, and `get_roi_as_dask` methods.
- Table ops moved to `ngio.images`
- int `label` as an explicit attribute in `Roi` objects (previously only in stored in name and relying on convention)
- Slight changes to `Image` and `Label` objects. Some minor attributes have been renamed for consistency.

### Table specs

- Add `t_second` and `len_t_second` to ROI tables and masking ROI tables

## [v0.3.5]

- Remove path normalization for images in wells. While the spec requires paths to be alphanumeric, this patch removes the normalization to allow for arbitrary image paths.

## [v0.3.4]

- allow to write as `anndata_v1` for backward compatibility with older ngio versions.

## [v0.3.3]

### Chores

- improve dataset download process and streamline the CI workflows

## [v0.3.2]

### API Changes

- change table backend default to `anndata_v1` for backward compatibility. This will be chaanged again when ngio `v0.2.x` is no longer supported.

### Fix

- fix [#13](https://github.com/BioVisionCenter/fractal-converters-tools/issues/13) (converters tools)
- fix [#88](https://github.com/BioVisionCenter/ngio/issues/88)
