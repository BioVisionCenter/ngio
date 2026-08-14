# Changelog

## [Unreleased]

### Fixes

- **ngio no longer warns you to pass a `mode` you cannot pass.** The `consolidate` deprecation notice fired from internal call sites — `create_ome_zarr_from_array`, `create_synthetic_ome_zarr`, and the writing iterators' `post_consolidate()` — pointing at the caller's frame with advice about a kwarg those functions did not expose. Creation and the iterators now expose `consolidation_mode` and forward it, so the notice is actionable where it appears; the synthetic helper pins `mode="dask"` and stays silent. The gap was invisible in CI because the test configuration ignores `NgioFutureWarning` globally.
- **The metadata memo can no longer pair fresh attributes with a stale decode under threads.** `_MetaMemo` stored its two halves in separate assignments, so a reader landing between them saw the new attributes with the old metadata — and the plate fan-out shares one handler across worker threads. The pair is now a single atomic snapshot, and the redundant defensive copy of the attributes went with it (`load_attrs` already hands out a private dict).
- **Concurrent fan-out threads now share one cached object per well, image and child handler.** `_get_well`, `_get_image` and `get_handler` inserted with check-then-act, so two racing threads could each keep their own object — breaking the `get_well_images` identity guarantee, and leaving a group handler outside `invalidate_meta`'s cascade to serve stale metadata. All three now insert atomically and every racer gets the winner.
- **`refresh()` now reaches live `Label` objects.** Labels do not share the images' meta handler — each carries its own — so a `Label` held across `refresh()` kept serving its construction-time `dimensions`, even under `cache=False`, despite the docstring's promise. The labels container now keeps one meta handler per label (which also drops a redundant decode from every `get_label`) and `refresh()` invalidates them. `OmeZarrPlate.refresh()` additionally resets its tables container, which carries the per-table type memo, matching what the image container already did.
- **A well at a different NGFF version than its plate decodes again.** Handing the plate's resolved version down to its wells turned the version into a constraint: a well rewritten at another version raised `NgioValidationError` where the registry walk used to decode it. The supplied version is now a fast path — tried first, with the rest of the registry as fallback — so the optimisation keeps the tolerance.
- **Clipping shards to a level's shape can no longer produce a geometry zarr rejects.** `PyramidLevel` clipped chunks and shards to the shape independently, so `shape=(1, 15)`, `chunks=(1, 10)`, `shards=(1, 20)` came out as shards `(1, 15)` — not a whole multiple of the chunk, which `create_array` refuses. The clipped shard now rounds down to a chunk multiple, never below one chunk.
- `store_dask` rejects stepped slices in `region` — its single-writer-per-unit argument assumes a contiguous cut of the unit grid — and its `PerformanceWarning` suppression now matches the specific rechunk message instead of the whole category, so dask's other performance warnings surface again.
- Coarsening upward now says so instead of dividing by zero inside dask. `mode="coarsen"` aggregates whole blocks, so it can only ever downsample — but consolidating from a middle level asks for the levels *above* the source too, and those edges are upsamples. The block factor came out 0 and the failure surfaced from `dask.array.routines.aligned_coarsen_chunks` as a bare divide-by-zero that named neither coarsening, nor the level, nor the alternative. It now raises `NgioValueError` naming the axis, both sizes, and the modes that can upsample.
- Listing tables by type could *create* a group. `TablesContainer.list(filter_types=...)` opened each table through a helper that defaults to creating what it cannot find, so a stale name left in the `tables` attribute made a read write an empty group in `r+`, or raise "cannot create a group in read only mode" in `r`. An unfiltered `list()` returned the same name harmlessly. A stale name is now also *skipped* by typed listings — `list(filter_types=...)` and `list_roi_tables()` return the tables that do exist instead of raising on the one that does not; a direct `get` of the dangling name still raises.
- **A dask write could lose updates when the target's write unit was larger than dask's `array.chunk-size` budget** — a 256 MiB shard against the 128 MiB default, say. `to_zarr` sizes its blocks to that budget, so it emitted blocks straddling shard boundaries and then wrote them with `lock=False`: several writers on one shard, no serialisation. ngio now raises the budget to one write unit for the duration of the call, which removes the case by construction and costs nothing when the unit is already smaller.
- Dask writes are no longer serialised behind a process-global lock. `DASK_STORE_LOCK` was added in v1.0.0 to stop two blocks losing an update on one chunk, but it could never keep that promise across processes — `SerializableLock` rebuilds from a per-process registry, so spawned workers each get their own lock and a 4-process write still loses updates. It also made two unrelated images in one process write one at a time. It is removed rather than repaired: writes now go through `da.to_zarr`, which cuts the input on the target's own write-unit grid so each unit has exactly one writer, and contention that cannot occur does not need a lock. Cross-process safety is unchanged and remains `atomic_*` and the file lock.
- **ngio's own writes could be invisible to ngio on a store carrying consolidated metadata.** zarr leaves `.zmetadata` untouched when attributes are written and ngio has no way to refresh someone else's, so every read came back from a snapshot taken before ngio's writes — a label ngio had just derived was absent from the very next `list_labels()`. ngio now ignores consolidated metadata everywhere. It never wrote any, so no store ngio produces changes.

### Behaviour changes

- **`cache=True` now actually caches metadata.** `ZarrGroupHandler.load_attrs` reopened the group unconditionally, so the flag was inert for every NGFF document and an outside write showed up whether or not you asked for caching. It now means what it says: metadata is held for the object's lifetime, and a write that goes around the handler is not visible until you call the new `refresh()`. `cache=False` is unchanged and remains the default, so nothing moves unless you were already passing `cache=True`.
- **`image.dimensions` is now fixed for the lifetime of the image object.** It was rebuilt on every access, which meant a full metadata reload — but `zarr_array` is fetched once in `__init__` and never refreshed, so `shape` and `chunks` were *already* a construction-time snapshot and only the dataset was re-read. Both halves are now consistent. A write through the image re-derives it; a write by another process is not seen until you reopen the container or call `refresh()`, which is the guarantee `zarr_array` already carried.
- **A malformed well now raises on first use rather than when the well is opened**, when the well is reached through its plate. Reaching one directly with `open_ome_zarr_well` is unchanged. The exception type and message are the same either way.
- **ngio no longer wraps a plain local store in `NgioStore` when the retry policy is a no-op.** `NgioStore` is a `WrapperStore`, and zarr's `create_codec_pipeline` silently falls back to its own `BatchedCodecPipeline` for any store the configured pipeline does not recognise — so the wrapper cost every user `zarrs` while, with the default `io_retry` (`max_retries=0`), being a pure pass-through in return. The wrapper is still attached whenever it changes behaviour: a configured `io_retry`, on Windows (where the sharing-violation retry is unconditional), for any non-local store, or when you pass an `NgioStore` yourself. Store services are unchanged — `ZarrGroupHandler.store` exposes the `NgioStore` facade either way, so `full_url`, `get_mapper` and `local_root` behave as before. Configure `io_retry` before opening a group; it is snapshotted at construction, as it already was.
- `OmeZarrPlate.get_well_images` now returns the same cached `OmeZarrContainer` objects as `get_images` and `get_image` under `cache=True`, instead of building fresh ones per call. Mutating one is visible to the next caller — which was already true of the other two, so this makes the three consistent.
- **A codec pipeline that silently fell back now warns.** zarrs rejects wrapper stores, `MemoryStore`, every `FsspecStore` including S3, and even `LocalStore` subclasses, by raising `NotImplementedError` — which zarr-python catches and ignores while `zarr.config` keeps reporting the configured pipeline. ngio now reports it once per process per store type, naming the concrete class.
- **The `dask[array]` floor is raised to 2025.11**, the first release whose `to_zarr` both rechunks the input to the target's write unit and cuts a region on that same grid. Below it, `to_zarr` writes with `lock=False` and no alignment — the shape of the data loss v1.0.0 added a lock to stop.
- **Caching and the atomic plate/well operations are no longer mutually exclusive.** `_create_lock` used to refuse outright when caching was on, because a read-modify-write could otherwise be served a value cached from before the lock was taken. Taking the lock now refreshes cached metadata on entry and again on release, so the hazard is handled rather than forbidden — `atomic_add_image` and friends work with `cache=True`. Verified by a 40-item, 4-process no-lost-update test that fails without the invalidation.

- **`MapperProtocol` changed shape — a clean break, no deprecation period.** A mapper now receives `(func, units)` where each `IterUnit` bundles one ROI's index, the ROI, its getter and its setter (`None` for a read-only unit), and returns the results in ROI order instead of `None`. The old `(func, getters, setters)` protocol could only write — a read-only iterator was an error rather than a use case — and gave a parallel implementation no identity to schedule or order by. No known consumer implements a custom mapper (none in ngio, its tests, its docs, or downstream); `BasicMapper` and every `map_as_*` call site keep working unchanged. Taken without a shim because a structural protocol cannot support two shapes at once.
- **`check_if_chunks_overlap` now measures the write target, at write granularity.** It used to build chunk sets from the *input* image's chunk grid — but writes land on the output, whose chunking can differ, and on a sharded output the atomic unit is the shard, not the chunk. It now computes each ROI's write footprint from the setters (`ChunkRect`, exact for slice/int selections) against `shards or chunks` of the actual output array; a read-only iterator has no write hazard and returns `False`. Two regressions the old check missed are now pinned by tests: ROIs disjoint on the input grid but sharing a coarser output chunk, and ROIs disjoint at inner-chunk granularity but sharing one shard.

### Deprecated

- **`consolidate(mode=...)` will default to `"auto"` in `ngio=1.2`**, building a small pyramid in memory instead of through dask — 3–5x faster, at a peak of roughly 1.6x the source level rather than a chunk-bounded one. Calling `Image.consolidate()` or `Label.consolidate()` without a `mode` on a pyramid that `"auto"` would actually build differently now emits an `NgioFutureWarning`; it stays silent where `"auto"` would decline anyway, since a notice about a default that would not change your result is a notice you learn to filter out. Pass `mode="auto"` to opt in now, or `mode="dask"` to keep the current behaviour and silence it. `FutureWarning` rather than `DeprecationWarning` for the same reason as `max_workers` below.
- **`max_workers` will default to `"auto"` in `ngio=1.2`**, so plate-wide operations — `get_wells`, `get_images`, `images_paths`, `list_image_tables`, `concatenate_image_tables` — will read their items concurrently instead of one at a time. Results and their order are unchanged; only the concurrency is. Calling one of these with more than one item and no `max_workers` now emits an `NgioFutureWarning`. Pass `max_workers="auto"` to opt in now, or `max_workers=1` to keep reading serially. `FutureWarning` rather than `DeprecationWarning` because Python hides the latter from end users, which is the wrong audience for a silent behaviour change.

### Features

- **`consolidate(mode="auto")` picks the in-memory path wherever it is provably identical to the chunked one.** `mode="numpy"` is 3–5x faster than `mode="dask"` but holds a whole level at once, and until now nothing chose between them for you. `"auto"` does, on two tests rather than one. Size, against `ngio.ConsolidationConfig.numpy_max_bytes` (default 256 MiB, measured on the *source* level — the chain never holds more than two adjacent levels, so peak is about 1.6x that; set it to `0` to disable the path entirely). And agreement: `dask` zooms per block with no halo, so it matches the whole-array zoom only for an integral-ratio downsample at `order` in `{"nearest", "linear"}`. Outside that envelope the two are different algorithms, not one algorithm at two block sizes — a ratio of 2.004, which is what a 501-pixel axis gives, puts them 99.9% apart, upsampling 1.6% apart, and `order="cubic"` 10–16% apart even on a plain power-of-two pyramid. `"auto"` declines silently in all of those and uses `dask`, so it can never trade an answer for speed. The default is still `"dask"`; see the deprecation note above.
- `OmeZarrContainer.refresh()` and `OmeZarrPlate.refresh()` re-read every piece of metadata the object is holding — the raw attributes cached under `cache=True`, the decoded metadata memo, and each image's `dimensions`. The last two are held regardless of the `cache` flag, so `refresh()` is not a no-op under `cache=False`.
- **Iterators gained `reduce_as_numpy`/`reduce_as_dask`**: apply a function to every ROI and collect the results in ROI order, writing nothing — the natural shape of per-ROI measurement, which previously had to go through a writing `map` or a hand-rolled loop. Units are built read-only even on writable iterators, and `post_consolidate` does not run.
- **`by_chunks` gained `grid: "read" | "write"`** (default `"read"`, today's behavior). `"write"` sizes the tiles by the *output* image's write granularity — the shard shape when the output is sharded, the chunk shape otherwise — so the resulting ROIs are collision-free on the write target by construction, falling back to the input grid for read-only iterators. `AbstractImage` gained the `write_granularity` property behind it.
- **`NgioConfig` gained a `zarr` section** (`ZarrConfig`), forwarding two knobs into zarr's own runtime configuration: `async_concurrency` — how many store requests zarr keeps in flight for one operation (zarr's default is 10, which fetches a 64-chunk read in ~7 serialized waves on a remote store; the concurrency gate pins that this knob genuinely widens and narrows the fan-out) — and `threading_max_workers`, the size of zarr's decode executor. Both default to `None`: the default ngio config leaves zarr's configuration byte-for-byte untouched. Applied during `import ngio` because zarr snapshots `threading.max_workers` into a process-global executor at first use; `async_concurrency` is read per call and can also be changed at runtime via `ngio.utils.apply_zarr_config` or `zarr.config.set` directly.
- Every `max_workers` argument accepts `"auto"`, which sizes a thread pool for round-trip-bound work (`min(32, cpu_count + 4)`) rather than making you pick a number. `OmeZarrPlate.images_paths` gained the argument, which it had no way to pass down before. The default stays `None` — serial — so nothing moves unless you ask. On a 96-well plate with 2 ms of simulated per-request latency, `images_paths()` goes from 1.21 s to 0.36 s with `max_workers="auto"`, and to 0.04 s combined with `cache=True`.

### Performance

- **Dask writes no longer assemble 128 MiB blocks in memory.** `da.to_zarr` glues whole write units into blocks sized to dask's `array.chunk-size`, which defaults to 128 MiB against a write unit that is typically a few hundred KiB — so it held roughly a thousand units resident per block, for nothing. The unit grid is what makes the write safe; the block grid is only batching, and peak memory is about the number of blocks in flight times their size. ngio now caps it, at **8 MiB** by default, configurable as `dask.write_block_max_bytes` (`null` defers to dask). Consolidating a 4 GB pyramid drops from **565 MB peak to 141 MB (−75%)** and a 2 GB one from 370 to 88 (−76%), at no cost in wall clock (19.16 s against 20.18 s at 4 GB) and 0.37% more tasks in the graph. The value was chosen by sweeping seven caps across three sizes rather than picked: below roughly 4 MiB the curve flattens onto the dask task graph itself, which no cap reaches, and 8 MiB sits a doubling above that while still leaving at least four write units per block on coarse chunk geometries. The cap is a ceiling only — it never raises an `array.chunk-size` you set lower yourself, and never takes the budget below one write unit, so the lost-update guard is untouched and a geometry whose unit already exceeds the cap is unaffected. Store operation counts are byte-for-byte identical; this is memory only.

- **A chunk that overhangs its axis no longer inflates the memory a dask write uses.** `store_dask` raises dask's `array.chunk-size` to fit one whole write unit, which is what stops `to_zarr` emitting blocks that straddle a unit and writing them with `lock=False`. It measured that unit from the *declared* chunk, but a chunk extent may exceed the array extent it spans — chunks `(1, 10, 2160, 2560)` over a single-z image is an ordinary OME-Zarr shape, where the declared chunk is 105 MiB and the largest chunk that can exist is 10.5 MiB. `normalize_chunks` clips to the shape before it sizes anything, so the surplus was capacity no write could ever use, and once the overhang is large enough to clear dask's own 128 MiB default it stopped being free: chunks `(1, 100, 2160, 2560)` on that array took the budget to 1,055 MiB and the block dask assembles in memory from **84 MiB to 791 MiB**. The unit is now measured as it can exist in the array. The write grid is unchanged, so every store-operation count is byte-for-byte identical — this is memory only.

- **`consolidate(mode="numpy")` no longer re-reads each level it just wrote.** Every mode built level *i+1*, wrote it, and then read it straight back off the store to build level *i+2* — while the array that produced it was still in hand. The numpy path now chains through memory instead, releasing each level once no later level reads from it. `consolidate_numpy` goes from **20 chunk reads and 660,827 bytes to 16 and 529,755**, and on a 1 GB pyramid it is ~8.5% faster for 0.83 s less CPU. Peak memory is unchanged; this mode already held a whole level by design. The other two modes still round-trip through the store on purpose: chaining *their* levels into one dask graph was implemented and reverted, because it bought ~1% of wall clock — the re-read was already overlapped with parallel work — while doubling peak memory and raising an already-untenable task count. At 256×256 chunks a dask graph costs one task and ~2.4 KB per chunk, so a 100 GB image is ~820k tasks and ~1.7 GB of graph before a byte is read, and fusing every level into one graph made that ~1.23M tasks and ~3 GB.
- **Coarsening no longer carries a float64 intermediate.** `da.coarsen(np.mean, ...)` promotes an integer source to float64, and that 4×-wide intermediate stayed alive until the store cast it back on write. The cast now happens inside the expression, which is what the store did anyway, so the output is unchanged and the intermediate disappears: peak memory for `mode="coarsen"` drops **3.1× on a 256 MB pyramid (654 → 211 MB)**, and it is the larger part of a 1512 → 740 MB improvement measured at 1 GB.
- **`OmeZarrPlate.well_images_paths` no longer grows with the square of a well's image count.** It resolved the image's prefix once per image, and each of those re-opened the well and read the whole plate document *twice* — to rebuild a string it already had. It now resolves the well path and the well once. On a well with six images: **75 → 9 metadata reads**, 27 → 3 group reopens, 7 → 1 well opens, and 11,284 → 1,312 bytes read. `get_image` and `get_image_acquisition_id` are halved by the same fix to `_image_path`.
- `OmeZarrPlate.get_well_images` no longer rebuilds a container per call. It duplicated `get_image` minus the image cache, so under `cache=True` it handed back a different object than `get_images` did for the same path. Combined with the fix above, a six-image well goes from **117 → 51 metadata reads** and 39 → 15 group reopens.
- Selecting channels by label cost one full metadata reload **per channel**, on the hot path behind every `get_array`, `get_roi`, `set_array` and `set_roi`. The channel metadata is now read once per call: two channels drop from 3 metadata reads to 2, and the count no longer scales with the selection. Reads that pass `channel_selection=None` were already free and are unchanged.
- **The channel metadata is now also flat in the *call* count**, cached against the metadata generation exactly like `dimensions`. A loop of channel-selecting reads on one image — the ordinary shape of per-ROI processing — paid a metadata reload per iteration; now only the first one derives it, and any channel write re-derives it through the generation counter. The new `read_channel_selection_x3` scenario records it: three reads cost 2 metadata reads, the same as one, where each repeat used to add its own.
- **ROI tables no longer rebuild their DataFrame on every property access.** `table_data` re-iterated every ROI into a fresh frame per read — and tested for emptiness by materialising the full ROI list first — so a masking table with tens of thousands of labels paid hundreds of milliseconds per `.dataframe` touch. The rebuild now runs only after an `add()` has actually changed the ROIs, via a dirty flag.
- **`_check_for_mixed_types` no longer scans every cell of every column.** It ran a Python-level `.apply(type)` over each column on every pandas → anndata conversion, including `float64`/`int64` columns whose dtype already guarantees homogeneity. Concrete dtypes now return immediately, and categoricals scan their (small) category set instead of the values: a 200k-row numeric column drops from a full pass to ~14 µs.
- **`normalize_anndata` no longer deep-copies the whole AnnData to swap `obs`.** Fixing the index duplicated the entire `X` matrix in memory on every load whose index differs from what is on disk. It now builds a new AnnData around the same `X`; the components are shared, not copied.
- **The overlap checks are no longer quadratic.** `check_if_regions_overlap` sorts on the most discriminating axis and sweeps, comparing each region only against those still open — 2,048 disjoint ROIs drop from **1,083 ms to 31 ms**, and 8,192 finish in 0.3 s where the all-pairs scan would need ~17 s. `check_if_chunks_overlap` now accumulates one seen-set instead of intersecting every pair. List selections and stepped slices, which are not intervals, keep the pairwise fallback.
- **Reading an AnnData table probes fewer store keys.** The reader listed the group to learn whether the store can list at all, threw the listing away, and then probed all nine AnnData elements — plus a legacy `raw.X` membership test and a re-fetch of `obs` it had already read. The one listing now also filters the elements to those that exist and answers the compat probes. `table_load_anndata` and `table_load_roi` drop from **55 to 49 metadata reads**, and `plate_concatenate_tables` from 278 to 254.
- `OmeZarrContainer.__repr__` listed labels and tables twice each — four attrs reads per REPL echo. Bound once.
- `ZarrGroupHandler` gained a `path` property, so the table backends and `full_url` stop reopening the group — a full metadata read — to obtain a path that is fixed at construction. Loading a parquet table drops from **18 to 16 metadata reads** and 6 to 5 group reopens. The anndata read path already avoided it.
- **`zarrs` now actually engages on local stores**, where before it was configured and silently ignored. Installing `zarrs` and setting `codec_pipeline.path` is enough: on a 32 MB image with 256×256 chunks, `set_array` goes from **76.3 ms to 16.7 ms (4.6x)** and `get_as_numpy` from **39.2 ms to 9.4 ms (4.2x)**. It has to land after the write-alignment fix above, not before — a codec pipeline cannot help a path whose cost is whole-shard read-modify-writes, and pushing each of those through the Rust boundary is slower than leaving it alone.
- **Writing a dask array to a sharded image no longer read-modify-writes every shard.** The write unit of a zarr array is `shards or chunks`, never `chunks` alone: zarr can only skip the read-modify-write when a write covers a *whole* unit. `da.store` issues one `__setitem__` per dask block, so a block covering one of a shard's 64 inner chunks made zarr read, decode, merge, re-encode and rewrite the entire shard. Writes now go through `da.to_zarr`, which rechunks the input onto that grid first. Writing a 1 MB sharded image goes from **128 chunk reads and 33.2 MB read to 2 reads and nothing** — the two remaining reads are misses probing shards that do not exist yet — and from 128 chunk writes / 34.2 MB to **2 / 1.05 MB**. A region write drops from 96 reads and 18.6 MB to the same 2 and nothing. Unsharded images, which are the default, are byte-for-byte unchanged.
- Pyramid consolidation no longer rechunks to `target.chunks` before writing. On a sharded array that is the shard's *inner* chunk shape, so every block was a partial shard write — the rechunk was worse than doing nothing. `da.to_zarr` derives the right unit itself, so the explicit rechunk is gone from both the zoom and the coarsen path: `consolidate_sharded_dask` goes from **40 chunk writes and 2.40 MB written to 4 and 328 KB**, and reads 1.46 MB instead of 3.53 MB. The unsharded `consolidate_dask` is unchanged.
- Creating a container from an array is **96 → 25 metadata reads** and 26 → 6 group reopens. Two causes. `set_channel_windows` re-read `channels_meta` inside its own loop when the identical value was already bound a few lines above, and reading that property opens the image and reloads its metadata — one line, 11 reads. The rest is that writing the array, building the pyramid and computing the channel windows re-read the same handful of documents a dozen times over; that sequence now runs through a cached view, as `create_empty_plate` and `create_empty_well` already did. The container handed back is uncached, unchanged.
- Listing tables by type no longer walks every table once per type. The type lives in each table's own attributes, never in the `/tables` group, so filtering costs one group open per table — and `list_roi_tables()` asked for two types, reading every document twice to sort names the first pass had already sorted. It is now one pass, and the result is memoised, so on six tables it drops from **95 store reads to 51** cold and to **2** on a repeat. Three calls in a row go from 187 reads to 41.
- Answering "this image has no tables" was the *most* expensive listing there is, because the failed probe for the `/tables` group was never remembered. Three `list_tables()` calls on such an image go from **15 reads to 5**, and stay flat however many times you ask.
- `TablesContainer.get` read each table's attributes twice — once to learn the type, once to build the concrete metadata model. Now read once: `get_table` drops from 18 reads to 16, `table_load_anndata` 57 → 55, and `plate_concatenate_tables` 294 → 286. The module-level `open_table` had the same double read and got the same fix.
- Deriving `image.dimensions` once instead of per access removes the last metadata reload from the read path. Repeated access drops from **520us to 2.1us** on NGFF 0.4 and 379us to 1.3us on 0.5. In store-op terms `dimensions_x10` goes from 10 metadata reads and 24,320 bytes to **1 and 2,432**, and `read_rois` to **zero** metadata reads. The property is read by every `get_*`/`set_*`, once per ROI by every iterator, and twice per ROI by masked ones — so a 1000-ROI masked segmentation goes from roughly 2,000 metadata reloads before the first pixel to two.
- `create_empty_plate` no longer grows with the square of the plate. It looped `add_image`, and each iteration re-read the plate document, rewrote it whole, and rewrote the well document — so bytes written per image grew with the plate rather than staying flat. The plate document is now built in memory in one pass and each well document written once. For a 24-image plate: **733 → 178 metadata reads, 246 → 102 writes, and 82,074 → 8,870 bytes written**. The gap widens with size — a 384-image plate went from 14.4 MB written to 128 KB, and bytes-per-image is now flat (~334) instead of climbing from 1,528 to 37,490.
- Wells are told the NGFF version their plate already resolved, so a 0.5 plate stops paying a failed pydantic validation per well: walking 24 wells drops from 48 decode attempts (24 failed) to 24 (none failed).
- Opening a well no longer reads its metadata twice. `WellMetaHandler` read and decoded in its constructor to resolve the NGFF version — but when the caller already supplies the version, as a plate now does, that read only validated early, and a plate walking its wells was about to read every one of those documents again. The check moved to first use, where it raises the same error. `images_paths()` on a 24-well plate drops from **218 metadata reads to 170**, `get_wells()` from 170 to 122, and the attribute loads behind them halve (49 → 25). Opening a well directly, without a version, still validates in the constructor.
- Metadata reads under `cache=True` no longer touch the store at all. Combined with the decode memo below, **`image.dimensions` goes from 520us to 68us on NGFF 0.4 and 379us to 68us on 0.5** — 6–8x, on the property every iterator reads once per ROI and twice per masked ROI. In store-op terms `dimensions_x10` drops from 10 metadata reads and 24,320 bytes to **zero of both**, and walking a 24-well plate with `images_paths()` from 218 reads / 75,862 bytes to **72 / 2,904**. `cache=False` is unaffected and still reads the store every time.
- Metadata handlers no longer re-run the pydantic decode on every access. `get_meta()` still reads the group every call — so a change made by this process or another is picked up exactly as before — but it now decodes only when the raw attributes have actually moved. Decoding is ~20x the cost of copying the result (~700us against ~37us on a four-level image), and it was paid per access: `image.dimensions`, which every iterator touches once per ROI and twice per masked ROI, goes from **801us to 356us** on NGFF 0.4 and **717us to 268us** on 0.5. In store-op terms `dimensions_x10` and `read_rois` drop to **zero** decodes, `create_container_from_array` from 16 to 1, `plate_images_paths` from 49 to 24, and `create_plate` from 74 to 50.
- Consolidating a pyramid in the default `dask` mode read every source chunk twice. `compute_chunk_sizes()` ran immediately after an explicit `rechunk(target.chunks)`, executing the whole read → zoom graph purely to re-learn block shapes that rechunk had already fixed, then discarding the pixels for `da.store` to recompute. Halved: `consolidate_dask` goes from 40 chunk reads and 1,316,187 bytes to 20 and 660,827, and `create_container_from_array` from 48 and 1,419,229 to 28 and 763,869. Every writing iterator pays this through `post_consolidate()`.
- Plate and well metadata handlers now memoise the NGFF version they resolved, as the image handler already did. The decoder registry is 0.4-first, so a 0.5 plate previously paid a complete failed pydantic validation on *every* metadata read rather than once per handler. Walking a 24-well 0.5 plate with `images_paths()` drops from 98 decode attempts (49 failed) to 73 (24 failed).
- Dropping the consolidated-metadata probe also removes a `.zmetadata` read that missed on every ngio-written NGFF 0.4 store — one wasted round-trip per metadata reload, which matters most remotely. `plate_images_paths` on a 24-well plate goes from 291 metadata reads to 218, and `create_plate` from 733 to 514.

### Chores

- The performance gate gained `list_roi_tables`, `list_roi_tables_repeated` and `list_tables_absent_x3`, plus an `image_no_tables` fixture. The existing image fixture always builds a `/tables` group, so nothing in the suite could see the cost of answering "no tables" — the same blind spot the `plate_images_paths_v05` scenario was added for.
- The performance gate gained cached counterparts — `dimensions_x10_cached`, `plate_images_paths_cached` — so the metadata cache is measured rather than asserted. `open_container` and `open_container_cached` were byte-for-byte identical, which was the recorded form of the inert-flag bug; their inequality now holds the fix.
- The performance gate gained `plate_well_images_paths`, `plate_get_well_images`, `read_channel_selection` and `table_load_parquet`, plus `plate_multi_image` and `tables_parquet` fixtures. Every plate fixture had exactly one image per well, so no scenario could see a cost that grows *inside* a well; and the pyarrow backends were assumed unmeasurable, which is only true of their payload — the zarr metadata around the table is ordinary store IO.
- The performance gate gained `write_dask`, `write_sharded_dask`, `write_sharded_roi_dask` and `consolidate_sharded_dask`. Every fixture in the suite was unsharded, so nothing in it could see that a dask write to a *sharded* array read-modify-writes whole shards: writing a 1 MB array costs **128 chunk reads and 33.2 MB read** where the unsharded control reads nothing, and a region write costs 96 reads and 18.6 MB. Sharding is also the one layout where a partial write can silently lose an update, so these counters gate correctness as much as cost — zero store reads means no read-modify-write, on any machine at any worker count. The `*_sharded_*` scenarios are the likeliest to diverge across zarr versions first, since 3.3 reads shards through `get_ranges`; they agree on 3.1.6 and 3.2.1 today.
- **The performance gate gained a second instrument: deterministic concurrency assertions** (`tests/performance/test_concurrency.py`). Op counts are invariant to concurrency — a serial and a parallel `get_wells` tally identically — so the count gate is structurally blind to a parallelism regression, and the CHANGELOG's own "96-well plate, 2 ms latency" numbers above were measured ad hoc with nothing committed to keep them true. A rendezvous store now parks `get`/`set` on zarr's IO loop until `k` are in flight together, and a gauge records the maximum overlap: `get_wells(max_workers="auto")` must overlap all four wells, `list_image_tables` likewise, a serial control must never exceed one, and a 16-chunk read must overlap exactly `async.concurrency` fetches (and exactly 3 when that knob is lowered to 3). No wall-clock, no thresholds, xdist-safe; success costs nothing and a regression pays one bounded timeout before failing with the observed overlap.
- The performance gate gained an `iterator_map_numpy` scenario: a writing iterator end to end, recording per-ROI reads and writes, the per-ROI metadata probes, and `post_consolidate`'s whole-pyramid rebuild — the iterator knows exactly which regions it wrote and rebuilds every level anyway. A future region-scoped consolidation lands here as a committed `get.chunk`/`set.chunk` drop.
- The performance gate gained `read_channel_selection_x3` and `consolidate_auto` scenarios, a relational-invariant suite, and a truncation guard. The invariants (`test_invariants.py`) assert the *relationships* the baselines are supposed to encode — cached strictly cheaper than uncached, sharded writes reading no chunk data back, `auto` resolution costing nothing over the mode it picks, channel metadata flat in the call count — directly on the committed files, so a wholesale `--update-baseline` cannot silently destroy them as numeric churn. And `--update-baseline` now refuses to write when the run collected only a subset of scenarios: combined with `-k`/`--lf` it used to silently drop every unrun scenario from the JSON.
- The performance gate gained a `plate_images_paths_v05` scenario. Every plate fixture was NGFF 0.4, which is both the default and the first version the decoder registry tries — so no plate scenario could ever record a failed validation, and the version memo above was invisible to the gate. `meta.decode_fail` on the new scenario is what holds it.

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
