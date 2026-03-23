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
dev pre-commit run --all-files    # lint/format
dev ty .                          # type check (Ruff ty)
docs serve_docs                   # docs preview
test_nb                      # run notebooks
```

## Config
- Python: 3.11–3.14
- Ruff: line length 88, Google docstrings, `D401` ignored, docstrings off in `tests/`
- Versioning: VCS via `hatch-vcs` (git tags, no hardcoded versions)
- Coverage: branch coverage; omits `tests/`, `src/ngio/_version.py`
