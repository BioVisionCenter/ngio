# ngio

[![License](https://img.shields.io/pypi/l/ngio.svg?color=green)](https://github.com/BioVisionCenter/ngio/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ngio.svg?color=green)](https://pypi.org/project/ngio)
[![Python Version](https://img.shields.io/pypi/pyversions/ngio.svg?color=green)](https://python.org)
[![CI](https://github.com/BioVisionCenter/ngio/actions/workflows/ci.yml/badge.svg)](https://github.com/BioVisionCenter/ngio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BioVisionCenter/ngio/graph/badge.svg?token=FkmF26FZki)](https://codecov.io/gh/BioVisionCenter/ngio)

**A Python library for OME-Zarr bioimage analysis.**

ngio gives you an object-based API for [OME-Zarr](https://ngff.openmicroscopy.org/) — the
cloud-optimised format for large, multi-dimensional microscopy data. Open an image, reach
for the resolution level you need, work with labels, tables and regions of interest, and
scale the same code from one field of view to a whole plate.

## Installation

```bash
pip install ngio
```

or

```bash
mamba install -c conda-forge ngio
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

## Supported OME-Zarr versions

ngio supports OME-Zarr v0.4 and v0.5, backed by either Zarr v2 or v3 storage. Support for
v0.6 and later is planned.

## Versioning

ngio follows [semantic versioning](https://semver.org/): from 1.0 onwards the public API
is stable, and breaking changes are reserved for major releases.

## Documentation

Full documentation, including guides, tutorials and the API reference, is at
[biovisioncenter.github.io/ngio](https://biovisioncenter.github.io/ngio/). Every code
block in the docs is executed when the site is built, so what you read is what actually
runs.

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
