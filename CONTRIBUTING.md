# Contributing

Contributions are welcome! Please open an issue to discuss significant changes before opening a PR.

## Prerequisites

- [Pixi](https://pixi.sh) — manages all environments and dependencies
- Git

Install Pixi, then clone and set up:

```bash
git clone https://github.com/BioVisionCenter/ngio
cd ngio
pixi install
```

## Development

Work in the `dev` environment, which includes linters, type checker, and test dependencies:

```bash
pixi shell -e dev        # activate shell
# or prefix individual commands:
pixi run -e dev <cmd>
```

## Running tests

```bash
pixi run -e test pytest           # single run (Python 3.11)
pixi run -e test13 pytest         # a specific Python version: test11, test12, test13, test14
```

Coverage is reported automatically. Pull requests run a reduced matrix — Linux on Python 3.11 and 3.13 — while `main` and tags run the full one: `test11`–`test14` across Linux, macOS, and Windows.

## Linting and formatting

```bash
pixi run -e dev lint              # run all hooks on all files
```

This runs Ruff (lint + format), `typos` (spell check), YAML/TOML validation, and notebook output stripping. Hooks also run automatically on `git commit`.

## Commit conventions

Please follow [Conventional Commits](https://www.conventionalcommits.org/) — this is not enforced by a hook (yet), but helps maintain a clean history and enables automated changelog generation.

Examples:

```
feat: add support for multiscale labels
fix: correct axis order in NgffImage
docs: update contributing guide
```

## Opening a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes with tests where relevant.
3. Run `pixi run -e dev lint` and ensure all checks pass.
4. Open a PR against `main` with a clear description of what and why.

CI runs the linters and the test matrix automatically.

## Releasing *(maintainers only)*

First, update the changelog: `CHANGELOG.md` is maintained by hand, so rename its `## [Unreleased]` heading to `## [vX.Y.Z]` and commit that before tagging.

Versions are derived from git tags via `hatch-vcs`. Tag by hand and push:

```bash
git tag -a v1.2.0 -m "v1.2.0"     # or v1.2.0a1 / v1.2.0b1 for pre-releases
git push origin v1.2.0
```

Tag names are PEP 440 with a `v` prefix. Pre-releases count up manually (a1 → a2 → b1 → stable). Once the tag is pushed, CI checks it matches the built wheel version, builds, and publishes to PyPI. `deploy-docs` moves the `stable` alias for a plain `vX.Y.Z` tag, or publishes under the pre-release's own name.
