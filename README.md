<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/BioVisionCenter/ngio/main/docs/assets/logo-lockup-dark.svg">
    <img alt="ngio" width="320" src="https://raw.githubusercontent.com/BioVisionCenter/ngio/main/docs/assets/logo-lockup.svg">
  </picture>
</p>

[![License](https://img.shields.io/pypi/l/ngio.svg?color=green)](https://github.com/BioVisionCenter/ngio/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ngio.svg?color=green)](https://pypi.org/project/ngio)
[![Python Version](https://img.shields.io/pypi/pyversions/ngio.svg?color=green)](https://python.org)
[![CI](https://github.com/BioVisionCenter/ngio/actions/workflows/ci.yml/badge.svg)](https://github.com/BioVisionCenter/ngio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BioVisionCenter/ngio/graph/badge.svg?token=FkmF26FZki)](https://codecov.io/gh/BioVisionCenter/ngio)

**Next generation file format IO — a Python library for OME-Zarr bioimage analysis.**

ngio is built for [OME-Zarr](https://ngff.openmicroscopy.org/), a cloud-optimised format
that stores large, multi-dimensional microscopy images and their metadata in an efficient,
scalable way. It provides an object-based API for opening, exploring and manipulating
OME-Zarr images and high-content screening (HCS) plates, along with labels, tables and
regions of interest (ROIs) for extracting and analysing specific regions of your data.

## Installation

To install ngio, use whichever package manager you already work with — it is published on
both PyPI and conda-forge.

```bash
pip install ngio                    # pip
uv add ngio                         # uv project (or: uv pip install ngio)
pixi add ngio                       # pixi, from conda-forge (--pypi for PyPI)
mamba install -c conda-forge ngio   # mamba/conda
```

Then work through the
[quickstart](https://biovisioncenter.github.io/ngio/stable/getting_started/0_quickstart/).

## Key features

- **Object-based API** — open, explore and manipulate OME-Zarr images and HCS plates;
  derive new images and labels with minimal boilerplate.
- **Tables and ROIs** — tight integration with [tabular
  data](https://biovisioncenter.github.io/ngio/stable/table_specs/overview/), extensible
  table schemas, and measurements stored alongside the image.
- **Scalable processing** — iterators for building pipelines that generalise from a single
  ROI to a full plate, with a pluggable mapping mechanism for parallelisation.
- **Remote stores** — stream from S3 and other fsspec-backed sources, with a configurable
  IO retry policy.
- **Supported OME-Zarr versions** — ngio supports OME-Zarr v0.4 and v0.5, backed by either Zarr v2 or v3 storage. Support for
  v0.6 and later is planned.

## Versioning

ngio follows [semantic versioning](https://semver.org/): from 1.0 onwards the public API
is stable, and breaking changes are reserved for major releases.

## Documentation

Full documentation, including guides, tutorials and the API reference, is at
[biovisioncenter.github.io/ngio](https://biovisioncenter.github.io/ngio/). The worked
examples are executed when the site is built, so the code and the output you read are what
actually ran.

## Citing ngio

If ngio contributes to work you publish, please cite it. See
[`CITATION.cff`](https://github.com/BioVisionCenter/ngio/blob/main/CITATION.cff) for the
current citation metadata.

## Project

ngio is developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html),
University of Zurich, by [@lorenzocerrone](https://github.com/lorenzocerrone) and
[@jluethi](https://github.com/jluethi). It is released under the BSD-3-Clause
[licence](https://github.com/BioVisionCenter/ngio/blob/main/LICENSE), and developed in the
open on [GitHub](https://github.com/BioVisionCenter/ngio) — issues and contributions
welcome.
