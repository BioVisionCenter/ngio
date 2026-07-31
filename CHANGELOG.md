# Changelog

## [Unreleased]

### API Breaking Changes

Every API deprecated in `v0.5.0` (each warned "will be removed in `ngio=0.6`") is now removed — that release became `1.0.0`.

**Renamed members**

| Removed | Use instead |
| --- | --- |
| `OmeZarrContainer.image_meta` | `.meta` |
| `.levels_paths` on `OmeZarrContainer`, `ImagesContainer` | `.level_paths` |
| `.set_channel_percentiles(a, b)` on both | `.set_channel_windows_with_percentiles(percentiles=(a, b))` |

**Renamed / removed arguments**

| Removed argument | On | Use instead |
| --- | --- | --- |
| `version=` | `create_empty_plate`, `create_empty_well`, `derive_ome_zarr_plate`, `OmeZarrPlate.derive_plate` | `ngff_version=` |
| `check_type=` | `OmeZarrContainer.get_table`, `OmeZarrPlate.get_table` | `get_table_as(name, TableCls)` or `get_*_table(name)` |
| `labels=`, `pixel_size=` | all derive entry points | `channels_meta=`, `pixelsize=` |
| `xy_pixelsize=` | `create_empty_ome_zarr`, `create_ome_zarr_from_array` | `pixelsize=` |
| `xy_scaling_factor=`, `z_scaling_factor=` | same two | `scaling_factors=` |
| `channel_labels=`, `channel_wavelengths=`, `channel_colors=`, `channel_active=` | same two | `channels_meta=` |
| `labels=`, `wavelength_id=`, `start=`, `end=`, `percentiles=`, `colors=`, `active=`, `**omero_kwargs` | `set_channel_meta` on `OmeZarrContainer`, `ImagesContainer` | `channel_meta=ChannelsMeta(...)` |

Derive entry points: `OmeZarrContainer.derive_image`/`derive_label`, `ImagesContainer.derive`, `LabelsContainer.derive`, `derive_image_container`, `derive_label`, `abstract_derive`.

- **`pixelsize` is now required** on `create_empty_ome_zarr` and `create_ome_zarr_from_array`; omitting it raises `TypeError` instead of `NgioValueError: pixelsize must be provided.`.
- **Not affected**: the `pixel_size=` *lookup* argument on the getters (`get`, `get_image`, `get_label`, ...) stays.
- Internal: `_check_deprecated_scaling_factors` removed; `init_image_like` and `_compute_scaling_factors` lost their `yx_scaling_factor`/`z_scaling_factor` parameters.

**Behaviour changes**

- `OmeZarrContainer.derive_image` hardcoded `dtype="uint16"`, `dimension_separator="/"` and `compressors="auto"` while documenting "the value from the reference image will be used", so deriving from a `float32` image silently downcast it. All three are now `None` sentinels and inherit from the reference, matching `derive_label` and `derive_image_container`. `ImagesContainer.derive` carried the same three hardcoded defaults and is fixed with it. `derive_label` is unchanged: it still inherits from a reference `Label` and falls back to `uint32` only when the reference is an `Image`.
- `get_masked_label` resolved the masking label at the raw `pixel_size` argument rather than at the resolved label's pixel size, so `get_masked_label(path=...)` disagreed with `get_masked_image(path=...)`. It now mirrors `get_masked_image`.
- `list_roi_tables` returns `[]` on a container or plate with no tables, matching `list_tables`. It previously raised — `NgioValidationError` from `OmeZarrContainer`, `NgioValueError` from `OmeZarrPlate` — and could create the `tables` group as a side effect. Both classes now also return ROI tables in the same order (`roi_table` then `masking_roi_table`); `OmeZarrContainer` previously reversed it.
- `open_image` and `open_label` now default to `strict=False`, matching `get_image`, `get_label`, `get_masked_image`, `get_masked_label`, `ImagesContainer.get` and `LabelsContainer.get`. They were the only two entry points defaulting to `True`.
- `OmeZarrContainer.__init__` never forwarded `validate_paths` to its `ImagesContainer`, so passing `False` still validated every level. The flag is now forwarded and named `validate_arrays` everywhere, and defaults to `False` — opening a container no longer touches every pyramid level up front. `open_ome_zarr_container(validate_arrays=True)` restores the eager check. **Behaviour change**: a container whose multiscale metadata references a missing or malformed array now fails on first access to that level rather than at open.
- `ImplementedTableBackends.get_backend` no longer defaults `backend_name`. The default was dead — every caller passes it explicitly, and it named `"anndata"` while the real default is `"anndata_v1"`.

**Renamed, with deprecation warnings (removal in `ngio=1.1`)**

| Deprecated | Use instead |
| --- | --- |
| `conctatenate_tables` | `concatenate_tables` (the typo is fixed) |
| `set_axes_unit` on `AbstractImage`, `ImagesContainer` | `set_axes_units` |
| `levels_paths=` on `ImagePyramidBuilder.from_scaling_factors`/`from_shapes` | `level_paths=` |
| `validate_paths=` on `OmeZarrContainer`, `ImagesContainer` | `validate_arrays=` |
| `OmeZarrPlate.get_images_async`, `get_wells_async`, `images_paths_async`, `list_image_tables_async`, `concatenate_image_tables_async`, `concatenate_image_tables_as_async` | the sync method with `max_workers=` |
| `list_image_tables_async`, `concatenate_image_tables_async`, `concatenate_image_tables_as_async` in `ngio.images` | the sync function with `max_workers=` |

### Migration Guide (v0.5 → v1.0)

```python
# Renamed arguments
create_empty_plate(store, name="plate", version="0.4")        # before
create_empty_plate(store, name="plate", ngff_version="0.4")   # after

create_empty_ome_zarr(store, shape=(4, 64, 64), xy_pixelsize=0.5, z_scaling_factor=1.0)
create_empty_ome_zarr(
    store, shape=(4, 64, 64), pixelsize=0.5, scaling_factors=(1.0, 2.0, 2.0)
)

# Table type checking
table = ome_zarr.get_table("roi", check_type="roi_table")
table = ome_zarr.get_table_as("roi", RoiTable)   # or ome_zarr.get_roi_table("roi")

# Deriving
ome_zarr.derive_image(store, labels=["DAPI"], pixel_size=ps)
ome_zarr.derive_image(store, channels_meta=["DAPI"], pixelsize=(ps.y, ps.x))

# Iterators (the old path still works until 1.1, with a deprecation warning)
from ngio.experimental.iterators import SegmentationIterator
from ngio.iterators import SegmentationIterator   # or: from ngio import ...

# Async -> max_workers (the async forms still work until 1.1, with a warning)
images = asyncio.run(plate.get_images_async())
images = plate.get_images(max_workers=8)

paths = asyncio.run(plate.images_paths_async())
paths = plate.images_paths()          # never did IO worth parallelising

names = asyncio.run(list_image_tables_async(images))
names = list_image_tables(images, max_workers=8)

table = asyncio.run(concatenate_image_tables_async(images, extras=extras, name="t"))
table = concatenate_image_tables(images, extras=extras, name="t", max_workers=8)

# Channel metadata
ome_zarr.set_channel_meta(labels=["DAPI", "GFP"], wavelength_id=["A01", "A02"])
ome_zarr.set_channel_meta(
    channel_meta=ChannelsMeta.default_init(
        labels=["DAPI", "GFP"], wavelength_id=["A01", "A02"]
    )
)

create_empty_ome_zarr(store, shape=(2, 64, 64), axes_names=("c", "y", "x"),
                      pixelsize=0.5, channel_labels=["a", "b"])
create_empty_ome_zarr(store, shape=(2, 64, 64), axes_names=("c", "y", "x"),
                      pixelsize=0.5, channels_meta=["a", "b"])
```

`pixel_size=` only ever read `.y` and `.x`; `.z`/`.t` were ignored and z/time spacing came from the reference image. To override those too, pass `z_spacing=`/`time_spacing=` — new behaviour the old argument could not express.

### Packaging

- **Add `src/ngio/py.typed`.** The `Typing :: Typed` classifier has been declared since 0.x but the PEP 561 marker was missing, so every downstream type checker silently ignored ngio's annotations. It now ships in both the wheel and the sdist.
- `Development Status` moves from `3 - Alpha` to `5 - Production/Stable`, and `[tool.commitizen] major_version_zero` is now `false` so `cz bump` can propose a major version.
- **Drop the `requires-python` upper cap** (`>=3.11,<3.15` → `>=3.11`). ngio is pure Python; the cap only blocked installs on new interpreters.
- **Real dependency floors, every one of them tested.** Most declared dependencies had no lower bound at all, and `zarr>3` excluded zarr 3.0.0 itself without expressing what the code actually needs. Each floor below is installed and exercised by a `test-min-deps` CI leg. Several turned out to be forced by ngio's own dependencies rather than by ngio:

  | Package | Floor | Why |
  | --- | --- | --- |
  | `numpy` | `>=2.0` | zarr 3.1.6 requires numpy 2 |
  | `zarr` | `>=3.1.6` | `WrapperStore` and the `is_group_listable` behaviour |
  | `scipy` | `>=1.14` | first release supporting numpy 2 |
  | `fsspec` | `>=2025.3` | below this `NgioStore.sync_fs_and_path` returns an async filesystem, which zarr's `FsspecStore` rejects |
  | `anndata` | `>=0.12.5` | `settings.auto_shard_zarr_v3`, used by the AnnData backend |
  | `pydantic` | `>=2.11.5` | required by ome-zarr-models |
  | `pandas` | `>=2.2.2` | first 2.2.x supporting numpy 2 on Python 3.11 |
  | `ome-zarr-models` | `>=1.4` | `ValidTransform` |
  | `pyarrow` | `>=16` | 15 caps numpy<2 |
  | others | `polars>=1.0`, `dask[array]>=2024.1`, `pooch>=1.8`, `pillow>=10`, `filelock>=3.12`, `aiohttp>=3.9` | |

- **`pandas` is no longer capped below 3.0.** The suite passes unmodified against pandas 3.0.3 — including the new default `str` dtype, the arrow→pandas path and always-on copy-on-write — so `pandas>=1.2.0,<3.0.0` becomes `pandas>=2.2.2`.
- **`anndata`'s cap moves from `<0.13.0` to `<0.14.0`**; the suite passes against anndata 0.13.2. The cap stays because `_anndata_utils.py` imports six private anndata symbols and vendors a copy of its zarr reader. Note anndata 0.13 requires Python >=3.12, so Python 3.11 installs still resolve to 0.12.x.
- **Add an `s3` extra**: `pip install ngio[s3]`. The README advertises S3 streaming, but `s3fs` was not a declared dependency at all — it only arrived transitively via the `test` extra, so following the README raised `ImportError`. The floor is `s3fs>=2026.2.0`, the release that added `set_custom_error_handler`.
- **Drop unused hard dependencies.** `requests` and `distributed` are not imported anywhere in `src/`, `tests/` or `docs/`; `dask[array]` and `dask[distributed]` collapse to a single `dask[array]` entry.

### Enforcement

These are what stop the lists above from regrowing.

- **`ty` runs in CI and pre-commit, and `src/` is clean.** It was previously run nowhere, leaving ~120 diagnostics ungated despite the `Typing :: Typed` classifier. `src/` is now at zero and gated; `tests/` is deliberately not gated yet. Most fixes were real typing improvements rather than suppressions — `Table.from_handler`/`from_table_data` now return `Self` (so `get_as` and `open_table_as` are genuinely type-safe instead of carrying `# type: ignore[return-value]`), the metadata encoder/decoder registries are annotated, `MaskedSegmentationIterator._input` is narrowed to `MaskedImage`, and 9 stale `# type: ignore`s are gone. The remaining suppressions are third-party stub gaps (dask's `store`, zarr's fancy-index `__getitem__`, `AnnData.write_zarr`, pooch's `downloader`) and the four known `MaskedImage`/`MaskedLabel` LSP violations.
- **The docs build and every snippet run on pull requests.** `docs.yml` only triggered on pushes to `main` and tags, so a broken snippet or cross-reference failed only after merge — where it also blocked the dev-docs deploy. The new `docs` job lives in `ci.yml` so the deploy workflow's `contents: write` permission is not extended to PRs.
- **A `test-min-deps` CI leg installs the exact declared floors** from `.github/min-constraints.txt` and runs the suite, so the bounds are tested rather than asserted. `.github/check_min_deps.py` fails if that file drifts from `pyproject.toml`, if a floor is missing a pin, or if a pin did not take effect.
- **`deploy` is gated on every check**, not just the test matrix: `lint`, `check-manifest`, `typecheck`, `docs` and `test-min-deps` must all pass. It also runs under the `pypi` environment, runs `twine check dist/*`, and asserts the git tag matches the built wheel version before publishing.

### Fix

- **`import ngio` no longer fails when an old `s3fs` is installed.** `refresh_s3fs_config` runs at import of `ngio.utils._zarr_utils` and called `s3fs.set_custom_error_handler`, which only exists in s3fs 2026.2.0 and later — so any environment holding an older s3fs, for any reason, raised `AttributeError` on `import ngio`. s3fs is not an ngio dependency, so it now degrades instead: the custom handler is skipped, and an `NgioUserWarning` is emitted only if `s3fs.custom_retry_markers` was actually configured.

### Features
- The iterators graduated out of `ngio.experimental`: they now live in `ngio.iterators` and are re-exported from the top-level namespace, so `from ngio import SegmentationIterator` works. `ngio.experimental.iterators` still resolves the four classes but emits `NgioDeprecationWarning` on attribute access; it will be removed in `ngio=1.1`.
- Add a configurable IO retry policy: `NgioConfig.io_retry` (`RetryConfig`) with `max_retries` (default `0`), a backoff strategy (`ConstantBackoff`, `LinearBackoff`, `ExponentialBackoff`), and error matching via `retry_on` substrings or the discouraged blanket `retry_all_errors`. Ngio's own `NgioError`s are never retried. The public `ngio.utils.retry_io` decorator reads the global config at call time.
- Add `ngio.utils.NgioStore`, a picklable zarr `WrapperStore` that applies the `io_retry` policy to every store IO call and centralizes store-type dispatch (`store_type`, `full_url`, `sync_fs_and_path`, `get_mapper`, `local_root`, `memory_dict`, `list_dir_collected`, `from_any`/`ensure`). Every group ngio opens is now backed by it, so the policy covers metadata, pixel data, and lazy dask IO on workers. `ZipStore` is now explicitly supported (it previously warned).
- Apply `io_retry` to the IO paths that bypass the zarr store: the pyarrow backend's dataset load/write, the AnnData backend's direct local/fsspec writes, and the `fractal_fsspec_store` metadata probe (401s become `NgioValueError` inside the retried call, so they are never retried).
- **Parallelism is now a `max_workers=` argument on the sync API** rather than a separate async surface. `OmeZarrPlate.get_images`, `get_wells`, `list_image_tables`, `concatenate_image_tables`, `concatenate_image_tables_as` and the matching `ngio.images` functions all take `max_workers: int | None = None`; `None` keeps the current serial behaviour. The `_async` counterparts still work but warn, and will be removed in `ngio=1.1`. Note that the async path was never unbounded: `asyncio.to_thread` uses Python's default executor, capped at `min(32, cpu_count + 4)`. `max_workers` exists so the concurrency can be lowered for rate-limited stores and so ngio's IO stops sharing that process-wide pool.
- Concatenation no longer has two divergent implementations: the sync path now materializes each table (`.dataframe` or `.lazy_frame` per `mode`) before concatenating, which only the async path used to do. Sync and async now produce identical tables with identical laziness.
- **Public namespace.** `ngio` now exports `MaskedImage`, `MaskedLabel`, `Channel`, `S3FSConfig`, `derive_ome_zarr_plate`, `BasicMapper`, `MapperProtocol`, `get_sample_info`, `__version__`, the six `get_ngio_*_meta` readers (only the `update_*` writers were exported before), and all seven error classes. `AbstractImage` is now importable from `ngio.images`, and `AbstractBaseTable`, `ImplementedTables` and `write_table` from `ngio.tables` — so a third-party table type can be registered without reaching into private modules. Table classes stay in `ngio.tables` and are deliberately not lifted to the top level. `ngio.resources.__all__` was missing `get_sample_info`, its only callable, and `ngio.ome_zarr_meta.__all__` listed `NgffVersions` and `PlateMetaHandler` twice.
- **Error hierarchy.** `NgioTableValidationError` was the only ngio error that did not subclass a builtin, so `except ValueError` caught its siblings but not it; it now subclasses `NgioValidationError`. New `NgioKeyError` (subclassing `KeyError`, with a `__str__` that does not repr-quote the message). The `NgioValidationError` (data read off disk failed a spec check) versus `NgioValueError` (a caller argument failed a run time check) boundary is now written into their docstrings.
- `pixel_size` and `pixelsize` are deliberately kept distinct and are now documented as such: `pixel_size` (a `PixelSize`) is a *lookup key* that selects a pyramid level on the getters; `pixelsize` (a float or `(y, x)` tuple) is a *value* written by the create/derive entry points.
- Add `ngio.utils.deprecated_alias` and `ngio.utils.deprecated`, the decorators behind ngio's own deprecations. `deprecated_alias(old="new")` forwards a renamed keyword and raises `NgioValueError` if both spellings are passed; `deprecated(replacement=...)` warns when a callable is invoked. Both emit `NgioDeprecationWarning` naming the removal version (default `1.1`) and report the caller's stack frame; on `async def` callables the warning fires when the coroutine is created, not when it is awaited.

### Fix
- `add_table` and `write_table` now preserve the input table's backend instead of rewriting with `anndata_v1` ([#207](https://github.com/BioVisionCenter/ngio/issues/207)); `backend` defaults to `None`, meaning "use the table's own". **Behavior change**: copying a `parquet`/`csv`/`json` table via `get_table` + `add_table` now keeps that backend.
- `concatenate_image_tables` (and variants) built a wrong index: the name was never set, `mode="lazy"` duplicated values and `mode="eager"` hashed the original index. Both now produce the same unique per-row index, and the async variant forwards `mode` so lazy prefetching stays lazy.
- `Roi.union`/`Roi.intersection` dropped ROI name `""` and label `0` (truthiness instead of `None` checks); joined slices now keep a deterministic axis order.
- `Roi.from_values` now validates its inputs — it used `model_construct`, which skips pydantic validation entirely.
- `PixelSize.__eq__` raised `TypeError` against non-`PixelSize` objects; it now returns `NotImplemented`. **Behavior change**: pixel sizes with different `time_unit`s always compare unequal.
- `NgioWellMeta.add_image`/`remove_image` and `NgioPlateMeta.add_well`/`add_acquisition`/`remove_well` mutated the receiver in place — the returned "copy" shared its lists with the original.
- `AxesSetup.from_ordered_list` silently dropped a non-canonical axis when a canonical name appeared to its left (e.g. `["z", "custom", "y", "x"]`).
- `ngio.iterators` grid helpers: `grid()` gave every ROI the same name (now unique per tile, e.g. `t0_z0_y32_x64`), and `by_chunks` with overlap ≥ chunk size raises `NgioValueError` instead of `range() arg 3 must not be zero`.
- `get_config()` now builds the global config singleton lazily on first call instead of at `ngio.config` import time. Note that `import ngio` still materializes it (`ngio.utils._zarr_utils` applies the s3fs config at module scope), so `NGIO_CONFIG_PATH` must still be set **before** importing ngio — see `docs/getting_started/7_configuration.md`.
- The v0.5 metadata decoder discarded the normalized value for non-string axis `unit`s.
- Accessing labels on a read-only image without a `labels` group raised `NgioValueError` instead of degrading gracefully (`list_labels()` → `[]`, `labels_container` → `NgioValidationError`).
- An empty `RoiTable` with no backend was unusable: `.rois()`/`.add(roi)` raised instead of treating the table as empty.
- Duplicate ROI names did not survive a write/read roundtrip — the dedup renamed only the internal dict key.
- `OmeZarrPlate.get_well` never returned the instance it cached; repeated calls now return the cached one, matching `get_image`.
- A negative index inside a slicing sequence (e.g. `get_array(y=[-1, 0])`) raised a bare `AssertionError` (which vanishes under `python -O`); it now raises `NgioValueError`.
- Fix the broken run link in the scheduled-CI failure issue: `{{ repo }}` is a context object, so the URL rendered as `[object Object]`. Now uses `{{ repo.owner }}/{{ repo.repo }}`.

### Tests
- Raise coverage from 91% to 95%: new unit tests for the v0.5 metadata codec, container/image error paths, ROI-table and plate edge cases, `NgioCache`, the `FeatureExtractorIterator` dask path, and io_pipes error branches.
- Enable `test_derive_from_legacy_images`, never collected due to a missing `test_` prefix; rename `test_fail_derive_singleton` → `test_pyramid_clamps_singleton_dimensions`.
- Register a `network` marker so `-m "not network"` runs offline, and move the Zenodo downloads from conftest import time into session fixtures — collection no longer blocks on the network.
- Speed up fixtures: `moto_s3_server` is session-scoped, and read-only consumers of the test images and Zenodo datasets share session-scoped copies instead of re-copying up to 126 MB per test.
- Drop `--cov` from the default `addopts` (~40% local overhead); CI passes it explicitly on the codecov leg.
- Deduplicate the 18-item test-image parametrization into a shared `zarr_name` fixture; merge the four per-backend round-trip tests into one parametrized test.
- Anchor test-data paths to `__file__` so pytest can run from any directory; reduce `test_multiprocessing_safety` from 1000 to 100 tasks.

### Chores
- Speed up CI: run with `pytest-xdist` (`-n 4`), collect coverage on the ubuntu/`test11` leg only, add `--durations=10`. The Zenodo download is guarded by a `FileLock` and no longer re-extracts an already-unzipped dataset.
- Fix the CI data-cache `restore-keys` fallback never matching (double quotes are literal inside a YAML block scalar), which made every PR job re-download ~160 MB. The key now hashes the dataset registry instead of `tests/conftest.py`.
- Centralize concrete-store dispatch behind `NgioStore` services, so the group handler, `copy_group`, `is_group_listable`, and the table backends no longer isinstance-check store types or reach into internals. Note: AnnData fsspec writes now use a synchronous clone of the store's filesystem.
- Shrink the store-matrix test payload from `(3, 5, 64, 64)`/3 levels to `(3, 2, 32, 32)`/2 levels — same monitored behaviors, far fewer round-trips through the mocked S3/HTTP stores.
- Slim the iterator tests: the two mutating tests copy only the image they write to, and run the full 9-axes matrix on v0.5 plus a v0.4 smoke subset instead of both versions × 9.
- `NgioDeprecationWarning` is kept in `ngio.utils`, but `import warnings` and its import are gone from `_plate.py`, `_ome_zarr_container.py`, `_image.py`, `_abstract_image.py`, and `_create_utils.py`. The private `_set_channel_meta`/`_set_channel_meta_legacy` pair collapsed into the public `set_channel_meta`.

### Documentation
- Rebuild the docs on [Zensical](https://zensical.org) instead of MkDocs + Material, and move every executed code block out of the Markdown into standalone scripts under `docs/snippets/`, included via `pymdownx.snippets` and run at build time. The five tutorial notebooks become Markdown pages, and CI builds with `--strict`.
- Apply a new design system to the docs and the README. The full theme in `docs/stylesheets/ngio.css`.
- Rewrite the copy to the design system's content conventions across every page, the nav and `llms.txt`, and add a landing page, a glossary, and API cross-references.
- Correct what the docs claim against what ngio actually does, most substantially on the table specification pages.
- Add `CODE_OF_CONDUCT.md` and `CITATION.cff`, and move `CONTRIBUTING.md` to the repository root, single-sourced into the docs so GitHub's community widgets pick them up.
- Add a "Configuration" getting-started page documenting the config file location (`~/.ngio/ngio_config.json` / `NGIO_CONFIG_PATH`), the `io_retry` policy (fields, backoff strategies, marker matching, snapshot-at-open vs read-at-call semantics), and its relationship to the lower-level `s3fs.custom_retry_markers` mechanism. `NgioConfig`, `RetryConfig`, and `get_config` are now listed in the top-level API reference.
- Point the iterator pages, the three iterator tutorials and the Getting Started iterators page at `ngio.iterators`, and drop the "Experimental API" warnings now that the iterators are part of the stable API.

### Fix
- `add_table` (`OmeZarrContainer`, `OmeZarrPlate`, `TablesContainer.add`) and `write_table` now preserve the input table's backend instead of always rewriting with the default `anndata_v1` backend ([#207](https://github.com/BioVisionCenter/ngio/issues/207)). The `backend` parameter now defaults to `None`, meaning "use the table's own backend" (`meta.backend`, which is `anndata_v1` for tables created in memory); pass a backend name explicitly to convert. Supporting changes: `Table.backend_name` now falls back to `meta.backend` instead of returning `None` for in-memory tables, `set_backend(backend=...)` can declare a backend preference (with early name validation and alias normalization) on a table not yet attached to a store, and passing `backend=None` no longer raises. **Behavior change**: copying a table stored with a non-default backend (e.g. `parquet`/`csv`/`json`) via `get_table` + `add_table` now keeps that backend on the destination.
- Fix the broken run link in the scheduled-CI failure issue (`.github/TEST_FAIL_TEMPLATE.md`). The template interpolated `{{ repo }}`, which is the `@actions/github` context object `{ owner, repo }` rather than a string, so `JasonEtco/create-an-issue` rendered it as `[object Object]` (e.g. `https://github.com/[object Object]/actions/runs/...`). It now uses `{{ repo.owner }}/{{ repo.repo }}`.

### Chores
- Centralize concrete-store dispatch behind `NgioStore` services: `ZarrGroupHandler.full_url`, the file-lock path resolution, `copy_group`'s fsspec fast path, `is_group_listable`, and the pyarrow/anndata table backends no longer isinstance-check store types or reach into store internals (`store.root`/`store.fs`/`store._store_dict`). The JSON table backend and table metadata writes now go through the handler's `load_attrs`/`write_attrs` instead of raw `.attrs` access. One behavior note: AnnData fsspec writes now use a synchronous clone of the store's filesystem (as the pyarrow paths already did) instead of the store's own, possibly async, filesystem instance.

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
