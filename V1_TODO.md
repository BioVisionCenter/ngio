# ngio 1.0 release checklist

Working list of what remains before tagging `v1.0.0`. Grouped by whether 1.0 freezes the
decision — the semver promise in the README means anything in "Blockers" or "API surface"
becomes expensive to change afterwards.

Not a public document: this is a maintainer checklist, kept at the repo root rather than
under `docs/` because every `.md` in `docs/` is published as a site page.

## Done

- [x] Remove every API deprecated in v0.5 that warned "will be removed in `ngio=0.6`",
      with an `### API Breaking Changes` table and a migration guide in the changelog.
- [x] Promote the iterators out of `ngio.experimental` into `ngio.iterators`, re-export them
      at top level, and leave `ngio.experimental.iterators` as a deprecation shim
      (scheduled for removal in 1.1).

---

## Blockers

Wrong or misleading at release time.

- [ ] **Add `src/ngio/py.typed`.** `pyproject.toml` declares the `Typing :: Typed`
      classifier but the marker file does not exist, so every downstream type checker
      silently ignores ngio's annotations. One empty file; hatchling picks it up.
- [ ] **`Development Status :: 3 - Alpha` → `5 - Production/Stable`** (`pyproject.toml`).
- [ ] **Set `major_version_zero = false`** (`[tool.commitizen]`). While it is `true`,
      commitizen treats the project as 0.x and will not propose a major bump.
- [ ] **Decide the config-loading story** and make the code, the changelog and
      `docs/getting_started/7_configuration.md` agree. `get_config()` is lazy, but
      `ngio.utils._zarr_utils` applies the s3fs config at module scope, so `import ngio`
      materializes the singleton and `NGIO_CONFIG_PATH` set afterwards is ignored. The
      docs and docstring now describe the real behaviour; the open choice is whether to
      make the s3fs refresh lazy so the env var can be set after import.
    <!-- COMMENT: No, NGIO_CONFIG_PATH should be set before import. But of course the config can be modified after import. -->
- [ ] **Fix dependency bounds** (`pyproject.toml`):
  - `zarr>3` excludes zarr 3.0.0 itself; the code needs `WrapperStore` and the
    `is_group_listable` fix relies on zarr ≥ 3.1.6 behaviour. Declare a real floor.
  - No lower bound at all on `numpy`, `filelock`, `scipy`, `fsspec`, `pydantic`,
    `aiohttp`, `dask`, `ome-zarr-models`, `pooch`, `polars`, `pyarrow`, `pillow`. At
    minimum `pydantic>=2` (the code is v2-only), `polars>=1.0`, `pyarrow>=15`,
    `numpy>=1.26`, plus a floor on the pre-1.0 `ome-zarr-models`.
  - `anndata<0.13.0` and `pandas<3.0.0` will both block installs soon after release.
  <!-- COMMENT: Carefully update all pins. Evaluate how much work would be to move to pandas>3.0.0 and if this is breaking -->
- [ ] **Drop unused hard dependencies**: `requests` and `distributed` are never imported
      anywhere in `src/`, `tests/` or `docs/`. Collapse `dask[array]` + `dask[distributed]`
      into a single entry.
- [ ] **Add an `s3` extra.** The README advertises S3 streaming and there is an `s3fs`
      config section, but `s3fs` is not in `[project.optional-dependencies]` — it only
      arrives transitively via the `test` extra, so following the README raises
      `ImportError`.
- [ ] **Rename the misspelled public export `conctatenate_tables`** →
      `concatenate_tables` (`ngio.images`). 1.0 freezes the typo otherwise.

## API surface — one-way doors

Freezing these is the point of 1.0.

- [ ] **`derive_image` ignores its own documented contract.** It hardcodes
      `dtype="uint16"` (and `dimension_separator`, `compressors`) while documenting
      "the value from the reference image will be used", so deriving from a `float32`
      image silently downcasts. `derive_label` and the underlying
      `derive_image_container` correctly use `None` sentinels.
    <!-- COMMENT: derive_label should default to dtype=uint32, any other should use the None sentinel -->
- [ ] **`get_masked_label` resolves the masking label at the wrong pixel size** — it
      passes the raw `pixel_size` argument where `get_masked_image` passes the resolved
      `image.pixel_size`, so with `path=` and no explicit pixel size the two disagree.
- [ ] **`list_tables` returns `[]` but `list_roi_tables` raises**, on both
      `OmeZarrContainer` and `OmeZarrPlate`. The same empty state also raises two
      different types (`NgioValidationError` vs `NgioValueError`), and `list_roi_tables`
      returns the same tables in reversed order on the two classes.
- [ ] **`strict` default flips** between the free functions (`open_image`/`open_label`,
      `True`) and every method form (`False`).
- [ ] **Pick one `pixel_size` story.** `pixel_size` is deprecated on the `derive*` paths
      in favour of `pixelsize`, but remains live on the getters (`get_image`, `get_label`,
      `open_image`, …) with a different type.
    <!-- COMMENT: homogenize all pixel_size to the new `pixelsize` standard. But under a deprecation warning for version 1.1.0 -->
- [ ] **Decide what is public.** Currently importable only from private modules:
      `MaskedImage`/`MaskedLabel` (the *return types* of `get_masked_image`/
      `get_masked_label`), `AbstractImage`, `AbstractBaseTable`, `write_table`,
      `Channel`, `S3FSConfig`, `derive_ome_zarr_plate`, `MapperProtocol`/`BasicMapper`,
      the `get_ngio_*_meta` getters (all six `update_*` setters are exported),
      `get_sample_info`. Also: no error classes and no table types are exported at top
      level, and `__version__` is missing from `__all__`.
    <!-- COMMENT: Only the table module should not be exported in top level. Everything else should be, not Abstract or Base classes. -->
- [ ] **Settle naming inconsistencies**: `set_axes_unit` vs `set_axes_units`;
      `open_ome_zarr_container` vs `create_empty_ome_zarr`; `validate_arrays` vs
      `validate_paths` (same flag, different name *and* default); `mode` meaning three
      unrelated things; `derive_image(ref_path=)` vs `derive_label(ref_image=)`;
      `ImagePyramidBuilder` still taking the deprecated `levels_paths` spelling.
    <!-- COMMENT: Settle them but don't break backwards compatibility. Add deprecation warnings for version 1.1.0. -->
- [ ] **Settle the exception hierarchy.** `NgioTableValidationError` is the only ngio
      error that does not also subclass a builtin, so `except ValueError` catches its
      siblings but not it. `NgioValidationError` and `NgioValueError` have identical bases
      and undocumented boundaries. There is no `NgioKeyError`.
- [ ] **`TableBackend` alias is inert**: the `| str` member makes the `Literal` useless
      for checking, and the default `"anndata_v1"` is not in the `Literal` at all.
    <!-- COMMENT: Literal there is only for helping type checkers and linters to autocomplete -->
- [ ] **Decide the async surface.** The only async entry points are 6 methods on
      `OmeZarrPlate`, all `asyncio.to_thread` + unbounded `asyncio.gather` (a 384-well
      plate fans out 384 threads — likely the cause of #172). Either complete the surface
      or drop them in favour of `max_workers=` on the sync methods. Adding `max_workers`
      later is non-breaking; having `_async` methods at all is not.
    <!-- COMMENT: Deprecate async methods in favor of `max_workers` on sync methods. But do not remove them, mark them deprecated and to be removed in a future version 1.1.0 -->

## Correctness / robustness

- [ ] Replace the 66 raw `ValueError`/`TypeError`/`KeyError`/`RuntimeError` raises in
      `src/` with typed ngio errors, so `except NgioError` works. Worst cluster:
      `images/_table_ops.py` (10, all in exported functions).
- [ ] Convert the 19 `assert` statements in data paths to real errors — they vanish
      under `python -O`.
    <!-- COMMENT: They are usually used for type checking, not for runtime errors. But if you think they should be errors let's change them raise -->
- [ ] Resolve the `MaskedImage`/`MaskedLabel` override incompatibility: `ty` reports 4
      `invalid-method-override` errors and 4 `# type: ignore`s already acknowledge it,
      which contradicts the docs' claim that every `Image` method works on them.
- [ ] Narrow the `except Exception` in the metadata version-probing loop
      (`ome_zarr_meta/_meta_handlers.py`) — it currently converts any decoder bug into
      "Failed to decode metadata".
- [ ] Document or close the `ZipStore` hole: the pyarrow/parquet backend raises
      `NotImplementedError` for both read and write, though `ZipStore` is advertised as
      supported.

## Enforcement — do these early

These are what stop the lists above from regrowing.

- [ ] **Run `ty` in CI** (and/or pre-commit). It is currently run nowhere, so its
      ~120 diagnostics are ungated despite the project claiming to be typed.
- [ ] **Build the docs on pull requests.** `docs.yml` has no `pull_request` trigger and
      `ci.yml` runs neither `test_snippets` nor `build_docs`, so a broken snippet or
      cross-reference only fails after merge to `main` — where it also blocks the
      dev-docs deploy.
- [ ] **Gate `deploy` on `lint` and `check-manifest`**, not just `test`. Consider
      `environment: pypi`, `twine check dist/*`, and asserting the tag matches the built
      version.
- [ ] **Make the docstring rules actually fire.** Ruff's `D1xx` treats every symbol in a
      `_`-prefixed module as private, and every implementation module here is
      `_`-prefixed — so missing-docstring is silently disabled package-wide. Public API
      docstring coverage is ~81%, and only ~69% style-complete.
- [ ] Add a minimum-dependency-version CI leg once the floors above exist, otherwise the
      declared bounds stay untested.

## Docs

- [ ] Document `open_ome_zarr_container` properly — the first function in every tutorial
      has a one-line docstring and no parameter documentation. Same for `open_table`,
      `open_table_as`, `open_tables_container`.
- [ ] Add docstrings to `NgioStore` (17 undocumented public methods), `NgioCache`,
      `ZoomTransform`, `FeatureTable`, `GenericRoiTable`, and `ngio.io_pipes` — the
      `io_pipes` API page currently renders ~40 bare signatures.
- [ ] Add API reference pages for `ngio.config` (the backoff strategies are used in the
      configuration guide but documented nowhere), `ngio.ome_zarr_meta`, `ngio.resources`,
      `ngio.tables.backends`, `ngio.tables.v1`.
- [ ] Fix the landing-page example: `get_table(...).get(...)` does not type-check because
      `get_table` returns the base `Table`. Either narrow/overload it or route the docs
      through `get_roi_table`.
- [ ] Sweep the copy-paste docstrings that will render verbatim: three classes documented
      as "Placeholder class for a label", `Image.consolidate` saying "the label", the
      `OmeZarrPlate` table methods saying "in the image", `v05/__init__.py` saying "v0.4",
      `rows`/`columns` saying "the number of" while returning lists.
- [ ] Add a code example to the README, and confirm the advertised conda-forge
      availability.
- [ ] Verify `Label`'s eight aliased data-access methods actually render in the API
      reference — they are class-body assignments to underscore-named functions and
      mkdocstrings filters `!^_`.

## Release mechanics

- [ ] Merge the duplicate `### Fix` and `### Chores` headings in `[Unreleased]` and
      retitle it to `## [v1.0.0]`.
- [ ] Populate `version:` and `date-released:` in `CITATION.cff` (the README points users
      there to cite ngio); consider a Zenodo DOI.
- [ ] Reconcile the copyright holder: `LICENSE` says "2023, Lorenzo Cerrone",
      `mkdocs.yml` says "2024-2026, BioVisionCenter UZH".
- [ ] Metadata polish: weak `description`, no `keywords`, `[project.urls]` missing
      Documentation/Issues/Changelog (so PyPI has no docs link), PEP 639 `license` form,
      missing science/bio-imaging classifiers, `requires-python` upper cap `<3.15`.
- [ ] Confirm every `/stable/` docs link resolves after the first `v*` tag deploys, and
      note that the `^v[0-9.]+$` stable regex means a release candidate deploys no docs
      version at all.
- [ ] Decide whether the published `dev` extra should stay — it pulls
      `napari`/`pyqt5`/`notebook`, so `pip install ngio[dev]` fails on 3.13/3.14.
- [ ] Add an sdist include/exclude; the sdist currently ships `brand/`, `tests/data/`
      and `docs/`.

## Deferred past 1.0

- Optional table-backend dependencies (#173) — 1.0 ships with the current dependency set.
- `PointsTable` (#177). Worth confirming first that the table-extension mechanism
  (`AbstractBaseTable`, `ImplementedTables`, `from_handler`) is exported and documented,
  since that part *is* frozen by 1.0.
- Object-detection iterators (#111).
- Splitting the large modules (`_ome_zarr_container.py`, `_plate.py`, `_image.py`,
  `_abstract_image.py`) — deprecation removal already shrank the first two.
