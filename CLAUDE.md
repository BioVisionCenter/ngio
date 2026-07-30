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

## Docs
Snippet mechanics and build traps: see `docs/CLAUDE.md`. Snippet scripts repeat their
imports so each rendered block stands alone (hence the `docs/snippets/**` ruff
per-file-ignores).

## Config
- Python: 3.11–3.14
- Versioning: VCS via `hatch-vcs` (git tags, no hardcoded versions)
- Coverage: `source = ["ngio"]`; line coverage only (branch coverage is not enabled)

## Code Style

- Ruff: line length 88, target py311
- Google-style docstrings, rendered by mkdocstrings/Griffe as Markdown (disabled for tests):
  - Inline code uses single backticks (`` `None` ``), never RST double backticks
  - Don't restate types in prose — they live in the signature (`channel: The channel to load.`, not `channel (int): ...`)
  - Sections: `Args`, `Returns`, `Raises`, `Example`, `Note`
  - One-line summary, blank line, then body
  - Terse: behavior and edge cases only, don't restate the signature
- Type checking via `ty`
- Internal modules prefixed with `_`
- Spell check via typos — false positives go in `[tool.typos.default.extend-words]` in `pyproject.toml`
- Pydantic v2: `@field_validator` before `@classmethod`

## Changelog

- Follow the format in `CHANGELOG.md`.
- **Always** update `CHANGELOG.md` when making code changes — add entries under the current `## [Unreleased]` section (or create one if missing).
