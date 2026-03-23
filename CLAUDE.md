# CLAUDE.md

## Project Overview

**ngio** is a Python library for working with OME-Zarr files (Open Microscopy Environment Zarr format) in bioimage analysis workflows. It provides an object-based API for large multi-dimensional microscopy images, supporting HCS plates, labels, tables, and ROIs.

## Environment & Package Management

The project uses **Pixi** (not pip/conda directly) for environment management. Most tasks require activating a specific pixi environment.

```bash
pixi install                          # Install all environments
pixi shell -e dev                     # Activate dev environment (Python 3.11)
```

## Common Commands

### Testing
```bash
pixi run -e dev pytest                          # Run all tests (dev env)
pixi run -e test11 pytest                       # Python 3.11
pixi run -e test12 pytest                       # Python 3.12
pixi run -e test13 pytest                       # Python 3.13
```

### Linting & Formatting
```bash
pixi run -e dev pre-commit run --all-files      # All pre-commit hooks
```

### Type Checking
```bash
pixi run -e dev ty .                                             # Type checker (Ruff's ty tool)
```

### Documentation
```bash
pixi run -e docs serve_docs                      # Local docs preview
pixi run -e docs test_nb                         # Execute notebooks
```

## Key Configuration Details

- **Python versions supported:** 3.11–3.14
- **Ruff line length:** 88, Google docstring convention, `D401` ignored
- **Docstring rules disabled** in `tests/`
- **Versioning:** VCS-based via `hatch-vcs` — version comes from git tags, no hardcoded version strings
- **Coverage:** Branch coverage enabled; omits `tests/` and `src/ngio/_version.py`
