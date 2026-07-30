# Changelog

## [Unreleased]

### Fix
- Fix the deprecated `yx_scaling_factor` shim assembling scaling factors in `(z, x, y)` order instead of the canonical `(z, y, x)`: an asymmetric tuple like `yx_scaling_factor=(2, 4)` silently produced swapped per-level y/x scales. The shim also now trims leading factors for shapes with fewer than 3 axes (a 2D image previously raised a length-mismatch error).
- Fix `concatenate_image_tables` (and variants) index construction: the concatenated table's index name was never set due to an inverted `None` check, and `mode="lazy"` built the index from the extras columns only — producing duplicated index values — while `mode="eager"` also hashed the original table index. Both modes now produce the same unique per-row index. The async variant also forwards `mode` to the per-image prefetch, so lazy prefetching no longer eagerly loads dataframes.
- Fix `Roi.union`/`Roi.intersection` dropping ROI name `""` and label `0` (the joins used truthiness instead of `None` checks; label `0` is a legal label). The joined ROI's slices now also keep a deterministic axis order (the receiver's axes first, then axes only present in the other ROI) instead of an arbitrary set order.
- `Roi.from_values` now validates its inputs (it used `model_construct`, which skips pydantic validation entirely), so invalid values such as a negative `label` or fewer than 2 axes raise instead of being silently accepted.
- Fix `PixelSize.__eq__` raising `TypeError` when compared with a non-`PixelSize` object (it now returns `NotImplemented`, so `ps == "x"` is `False` and `ps in [...]` works) and make the `time_unit` comparison symmetric: previously `a == b` and `b == a` could disagree when only one side had a `time_unit`. **Behavior change**: pixel sizes with different `time_unit`s now always compare unequal.
- Fix `NgioWellMeta.add_image`/`remove_image` and `NgioPlateMeta.add_well`/`add_acquisition`/`remove_well` mutating the receiver in place: the "updated copy" they returned shared its lists with the original object, so the original was silently modified too. All five now leave the receiver untouched.
- Fix `AxesSetup.from_ordered_list` silently dropping a non-canonical axis name when a canonical name appeared to its left (e.g. `["z", "custom", "y", "x"]`): the canonical name reclaimed the slot already assigned to the custom axis. Canonical names now always claim their own slot and non-canonical names fill the remaining slots right-aligned.
- Fix `ngio.experimental.iterators` grid helpers: `grid()` gave every generated ROI the same name (they now get unique per-tile names, e.g. `t0_z0_y32_x64`, prefixed with `base_name` when given), and `by_chunks` with an overlap equal to or larger than the chunk size now raises a clear `NgioValueError` instead of crashing with `range() arg 3 must not be zero`.
- The global config singleton is now loaded lazily on first `get_config()` call instead of at import time, so setting `NGIO_CONFIG_PATH` after importing ngio (but before first use) is honored.
- Fix the v0.5 metadata decoder passing a non-string axis `unit` through unchanged: the stringification computed for non-string JSON values was discarded (`unit=v05_axis.unit` instead of the normalized value).
- Fix accessing labels on a read-only image without a `labels` group: `labels_container` (and everything routed through it) raised `NgioValueError: Cannot create a group in read only mode` instead of degrading gracefully like the tables path (`list_labels()` → `[]`, `labels_container` → `NgioValidationError`).
- Fix an empty ROI table with no backend being unusable: `RoiTable().rois()` / `.add(roi)` raised `NgioValueError` instead of treating the table as empty (the `set_table_data` path already handled this case).
- Fix duplicate ROI names not surviving a write/read roundtrip: the dedup in `RoiDictWrapper` renamed only the internal dict key, so the serialized table still contained duplicate index values. The renamed ROI now carries the deduplicated name.
- Fix `OmeZarrPlate.get_well` never returning the instance it cached (it stored one `OmeZarrWell` and returned a second, freshly built one; repeated calls now return the cached instance, matching `get_image`).
- A negative index inside a slicing sequence (e.g. `get_array(y=[-1, 0])`) now raises `NgioValueError` instead of a bare `AssertionError` (which would also vanish under `python -O`).

### Tests
- Enable `test_derive_from_legacy_images` (`tests/unit/images/test_create.py`), which was never collected due to a missing `test_` prefix; it now runs for both NGFF versions with the level pairing and channel-axis handling corrected. Rename the misnamed `test_fail_derive_singleton` to `test_pyramid_clamps_singleton_dimensions`.
- Deduplicate the 18-item test-image parametrization (six verbatim copies) into a shared `zarr_name` fixture in `tests/conftest.py`.
- Register a `network` pytest marker and apply it to the two tests hitting raw.githubusercontent.com, so `-m "not network"` runs offline. The Zenodo dataset downloads moved from conftest import time into session fixtures — test collection no longer blocks on the network.
- Remove `--cov` from the default pytest `addopts` (local runs no longer pay ~40% coverage overhead); CI passes `--cov=ngio --cov-report=xml` explicitly on the codecov leg and `ci_upstream.yml` drops the now-unneeded `--no-cov`.
- Merge the four per-backend round-trip tests in `test_backends.py` into one parametrized test; drop the redundant middle-layer copy of the #207 backend-preservation test (still covered at the `write_table` and container layers).
- Anchor test-data paths to `__file__` instead of the current working directory (`tests/conftest.py`, `tests/create_test_data.py`, `test_unit_v04_utils.py`), so pytest can run from any directory. Reduce `test_multiprocessing_safety` from 1000 to 100 tasks and delete an unreferenced meta JSON fixture.
- Raise overall coverage from 91% to 95%: new unit tests for the v0.5 metadata codec (`test_unit_v05_utils.py`, mirroring the v0.4 file), container/image error paths and deprecated shims, ROI-table and plate edge cases, `NgioCache`, the `FeatureExtractorIterator` dask path, and io_pipes error branches. `hcs/_plate.py`, `images/_image.py`, `utils/_cache.py`, `experimental/iterators/_feature.py`, `io_pipes/_match_shape.py`, `_zoom_transform.py`, and `_ops_slices.py` are now at 100%.
- Speed up fixtures: the `moto_s3_server` fixture is now session-scoped (tests isolate via `random_zarr_path()`, and starting the server cost ~4s per S3 test); read-only consumers of `images_all_versions` and the Zenodo cardiomyocyte datasets now share session-scoped copies (`images_all_versions_readonly`, `cardiomyocyte_tiny_path_readonly`, `cardiomyocyte_small_mip_path_readonly`) instead of re-copying up to 126 MB per test, while mutating tests keep fresh per-test copies; `test_real_mask` copies only the single image it writes to; `test_table_ops.py` builds its two sample containers once per module.

### Chores
- Speed up CI: run the test suite with `pytest-xdist` (`-n 4`, new `test` extra dependency); collect coverage and upload to codecov on the ubuntu/`test11` leg only (one report per run, unchanged content); add `--durations=10` to CI runs for runtime observability. The Zenodo download in `tests/conftest.py` is now guarded by a `FileLock` (xdist workers on a cold cache) and no longer re-extracts an already-unzipped dataset (`re_unzip=False`).
- Shrink the store-matrix test payload (`tests/stores/utils.py`) from a `(3, 5, 64, 64)`/3-level image to `(3, 2, 32, 32)`/2 levels — same monitored behaviors (multi-channel z-stack pyramid, label, all four table backends, derive across the 4×4 store matrix), far fewer object round-trips through the mocked S3/HTTP stores (~75s → considerably less on CI).
- Slim the iterator tests: the two mutating tests (`test_segmentation_iterator`, `test_masked_segmentation_iterator`) now copy only the single image they write to (new `single_image_copy` fixture, replacing per-test copies of all 18 test images; the unused `images_all_versions` fixture is removed) and run the full 9-axes matrix on v0.5 plus a v0.4 smoke subset (`yx`, `tczyx`) instead of both versions × 9 — the iterators are version-agnostic and the v0.4 write path stays monitored by the creation/store tests. The two read-only iterator tests keep the full 18-image matrix.
- Fix the CI data-cache `restore-keys` fallback never matching: the fallback key was written inside a YAML block scalar with double quotes (`"Linux-data-"`), which are literal there, so a cache-key rotation caused every PR job to re-download the ~160 MB Zenodo datasets during the test step (~80s/job). The cache key now also hashes `src/ngio/utils/_datasets.py` (the dataset registry) instead of `tests/conftest.py`, so test-suite refactors no longer rotate the key (`ci.yml`, `ci_upstream.yml`).

### Feature
- Add a configurable IO retry policy: `NgioConfig.io_retry` (`RetryConfig`) with `max_retries` (default `0`, never retry), a backoff strategy (`ConstantBackoff`, `LinearBackoff`, or `ExponentialBackoff`, each with `delay_s`, `max_delay_s`, `jitter`), and error matching via `retry_on` substrings (matched against `"ExceptionName: message"`) or the discouraged blanket `retry_all_errors` flag, which emits an `NgioUserWarning` and is mutually exclusive with `retry_on`. Ngio's own `NgioError`s are never retried. Retry helpers live in `ngio.utils._retry`; the public `ngio.utils.retry_io` decorator reads the global config at call time.
- Add `ngio.utils.NgioStore`, a picklable zarr `WrapperStore` that applies the `io_retry` policy to every store IO call (the policy snapshot travels with the store into dask task graphs) and centralizes store-type dispatch behind uniform services (`store_type`, `full_url`, `sync_fs_and_path`, `get_mapper`, `local_root`, `memory_dict`, `list_dir_collected`, `NgioStore.from_any`/`ensure`). Every group ngio opens (`open_group_wrapper` / `ZarrGroupHandler`) is now backed by an `NgioStore`, so the `io_retry` policy applies to all zarr IO — metadata, pixel data, and lazy dask reads/writes executed on workers (the policy snapshot pickles with the store into the task graph). User-provided `zarr.Group`s are transparently reopened on a wrapped store; `ZipStore` is now an explicitly supported store type (it previously warned).
- Apply the `io_retry` policy to the IO paths that bypass the zarr store: the pyarrow table backend's dataset load/write, the AnnData backend's direct local/fsspec writes, and the `fractal_fsspec_store` metadata probe (where 401 auth failures are translated to `NgioValueError` inside the retried call, so they are never retried — even in blanket mode).

### Documentation
- Rebuild the docs on [Zensical](https://zensical.org) instead of MkDocs + Material, and move every executed code block out of the Markdown into standalone scripts under `docs/snippets/`, included via `pymdownx.snippets` and run at build time. The five tutorial notebooks become Markdown pages, and CI builds with `--strict` (a plain build exits 0 even when a code block raised).
- Apply the commissioned design system: the mark-G logo set, a contrast-audited teal palette and full theme layer in `docs/stylesheets/ngio.css`, and six explanatory figures authored as inline SVG so they follow the light/dark toggle. Site chrome — header, tabs and footer — now sits on its own tone, framing the page rather than bleeding into it.
- Rewrite the copy to the design system's content conventions: sentence case, British spelling, second person, and openers that lead with what the reader does. Covers every page, the nav and `llms.txt`, and adds a landing page, a 14-term glossary, "Next steps" footers and API cross-references.
- Add `CODE_OF_CONDUCT.md` and `CITATION.cff`, and move `CONTRIBUTING.md` to the repository root, single-sourced into the docs so GitHub's community widgets pick them up.
- Fix defects found along the way: Markdown extensions Zensical had silently dropped (declaring `markdown_extensions` replaces its defaults rather than extending them), YAML front matter invalidated by unquoted colons on six pages, `site_url` pointing at the clone URL, tables rendering as literal `|---|`, and per-page snippet state now that each page gets a fresh `markdown-exec` session.
- Proofreading pass over the refactored docs. Corrections: the `strict` parameter of `get_image`/`get_label` defaults to `False` (nearest level), not `True`, and the page said the opposite; `set_array` takes a numpy or dask array, not a dask delayed object; `RoiTable.get` returns one ROI, not all of them; `get_well_images` returns a dict, and omitting `acquisition` returns every image in the well rather than an empty one; the object model figure labelled `ImagesContainer` as a non-existent `MultiscaleImage`; `by_zyx` splits the time axis rather than iterating z-planes; retry records go to a `ngio:<module>` logger, which is not a child of `ngio`; and the "every code block is executed" claim in `index.md`, `README.md` and `llms.txt` is now scoped to the worked examples, 15 blocks being static.
- Correct the table specifications against what ngio actually writes: document how the `backend` attribute is named (the `{backend_name}_v{version}` convention, and the `experimental_{json,csv,parquet}_v1` legacy aliases kept for older tables), the AnnData backend casts to `float64` only for heterogeneous dtypes and stores booleans in `X` rather than `obs`, `instance_key` is always written for feature and masking ROI tables, `index_key`/`index_type` are omitted entirely for generic and condition tables, extra ROI columns do round-trip (via `Roi` extras) and the plate-location and index columns were missing from the optional list, only four of the five table types are auto-detected on read (`GenericRoiTable` is reachable only via `get_generic_roi_table`), condition tables were missing from the getting-started list, and the feature-type column lists are declarative only — ngio writes but never reads them. Also fixes the malformed JSON-backend directory tree, notes the Zarr v3 `zarr.json` layout, corrects the plate tree's column and image labels, and makes every spec block parse (trailing commas removed, fences retagged `json5` so the `//` comments highlight instead of reading as syntax errors).
- Flag `ngio.experimental.iterators` as experimental on the iterators guide and API page, retitle the `ngio.iterators` reference page and nav entry to the real module path, and add an executed example to the iterators page, which previously had no runnable code.
- Consistency pass: `>>>` prompts removed so every example is copy-pasteable, `OME-Zarr Container` no longer code-formatted as if it were a class, `get_array` presented as the mode-dispatch form rather than a back-compat shim, uniform "API reference" titles, table-spec page titles matched to their nav labels, numbered steps across all five tutorials, and tutorial snippet comments moved to second person and British spelling (they render into the page).
- `CONTRIBUTING.md`: `lint` and the `bump-*` tasks need `-e dev`; PRs run a reduced CI matrix, not the full one; sentence-case headings.
- Add a "Configuration" getting-started page documenting the config file location (`~/.ngio/ngio_config.json` / `NGIO_CONFIG_PATH`), the `io_retry` policy (fields, backoff strategies, marker matching, snapshot-at-open vs read-at-call semantics), and its relationship to the lower-level `s3fs.custom_retry_markers` mechanism. `NgioConfig`, `RetryConfig`, and `get_config` are now listed in the top-level API reference.

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
