# ngio
Python library for OME-Zarr files (bioimage analysis). Object-based API for multi-dimensional microscopy images; supports HCS plates, labels, tables, ROIs.

## Setup
Package manager: **Pixi** (not pip/conda)
```bash
pixi install          # install envs
pixi shell -e dev     # activate dev (Python 3.11)
```

## Commands
If the shell is activated, you can run commands directly. Otherwise, prefix with `pixi run -e <env> <command>`.
```bash
dev pytest                        # tests (dev)
test11 pytest                     # Python 3.11
test12 pytest                     # 3.12
test13 pytest                     # 3.13
dev lint                          # lint/format (pre-commit hooks)
dev ty check                      # type check (Ruff ty)
docs serve_docs                   # docs preview
docs build_docs                   # build the site
docs test_snippets                # run every docs snippet script standalone
docs clean_docs_data              # drop generated ./data/*.zarr stores
```

## Docs snippets
Executed code on the docs pages lives in real Python scripts under `docs/snippets/`,
not in the markdown. A page includes a named section via `pymdownx.snippets`:

````markdown
```python exec="true" source="material-block" session="get_started"
--8<-- "docs/snippets/getting_started/get_started.py:levels"
```
````

- Sections are delimited by `# --8<-- [start:name]` / `# --8<-- [end:name]`.
- Use `source="material-block"` (not `block`) and `html="1"` for figure blocks.
- One script per session; each must run standalone from the repo root.
- Sections repeat their imports so each rendered block stands alone (hence the
  `docs/snippets/**` ruff per-file-ignores).

## Config
- Python: 3.11–3.14
- Versioning: VCS via `hatch-vcs` (git tags, no hardcoded versions)
- Coverage: branch coverage; omits `tests/`, `src/ngio/_version.py`

## Code Style

- Ruff: line length 88, target py311
- Google-style docstrings, rendered by mkdocstrings/Griffe as Markdown (disabled for tests):
  - Inline code uses single backticks (`` `None` ``), never RST double backticks
  - Don't restate types in prose — they live in the signature (`channel: The channel to load.`, not `channel (int): ...`)
  - Sections: `Args`, `Returns`, `Raises`, `Example`, `Note`
  - One-line summary, blank line, then body
  - Code examples in fenced ` ```python ` blocks, not `>>>` doctests
  - Terse: behavior and edge cases only, don't restate the signature
- Type checking via `ty`
- Internal modules prefixed with `_`
- Spell check via typos — false positives go in `_typos.toml`
- Pydantic v2: `@field_validator` before `@classmethod`

## Changelog

- Follow the format in `CHANGELOG.md`
- **Always** update `CHANGELOG.md` when making code changes — add entries under the current `## [vX.Y.Z]` section (or create one if missing).
- Use these subsections (omit empty ones):
  - `### Features` — new user-visible behaviour
  - `### Fix` — bug fixes
  - `### API Breaking Changes` — anything that breaks existing call sites (include before/after example)
  - `### Chores` — internal refactors, dependency bumps, CI changes
  - `### Documentation` — doc-only changes
- One bullet per logical change; use backticks for identifiers.
