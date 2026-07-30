---
description: ngio is a Python library for OME-Zarr bioimage analysis, with an object-based API for images, labels, tables, ROIs and HCS plates.
---

# ngio

**Next generation file format IO — a Python library for OME-Zarr bioimage analysis.**

ngio is built for [OME-Zarr](https://ngff.openmicroscopy.org/), a cloud-optimised format
that stores large, multi-dimensional microscopy images and their metadata in an efficient,
scalable way. It provides an object-based API for opening, exploring and manipulating
OME-Zarr images and high-content screening (HCS) plates, along with labels, tables and
regions of interest (ROIs) for extracting and analysing specific regions of your data.

## Key features

- **Object-based API** — open, explore and manipulate OME-Zarr images and HCS
  plates; derive new images and labels with minimal boilerplate.
- **Tables and ROIs** — tight integration with [tabular
  data](table_specs/overview.md), extensible table schemas, and measurements stored
  alongside the image.
- **Scalable processing** — iterators for building pipelines that generalise from a
  single ROI to a full plate, with a pluggable mapping mechanism for parallelisation.
- **Remote stores** — stream from S3 and other fsspec-backed sources, with a
  [configurable IO retry policy](getting_started/7_configuration.md).
- **Supported OME-Zarr versions** — ngio supports OME-Zarr v0.4 and v0.5, backed by either Zarr v2 or v3 storage. Support for 
  v0.6 and later is planned.

## Installation

=== "pip"

    ```bash
    pip install ngio
    ```

=== "mamba/conda"

    ```bash
    mamba install -c conda-forge ngio
    ```

## ngio in 30 seconds

```python
from ngio import open_ome_zarr_container

# Open a container and inspect what is inside
ome_zarr = open_ome_zarr_container("path/to/image.zarr")
print(ome_zarr)  # levels, labels and tables at a glance

# Grab the highest-resolution image and read a channel as numpy
image = ome_zarr.get_image()
data = image.get_as_numpy(channel_selection="DAPI")

# Slice by a region of interest, in world coordinates
roi = ome_zarr.get_table("FOV_ROI_table").get("FOV_1")
patch = image.get_roi_as_numpy(roi)
```

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting started**

    ---

    Install ngio and work through the core objects: containers, images and labels,
    tables, masked images and HCS plates.

    [:octicons-arrow-right-24: Quickstart](getting_started/0_quickstart.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    End-to-end walkthroughs: create an OME-Zarr, process and segment images, extract
    features, and explore a plate.

    [:octicons-arrow-right-24: Browse tutorials](tutorials/create_ome_zarr.md)

-   :material-table:{ .lg .middle } **Table specifications**

    ---

    The on-disk spec for ROI, masking ROI, feature, condition and generic tables, and
    the backends that store them.

    [:octicons-arrow-right-24: Read the spec](table_specs/overview.md)

-   :material-api:{ .lg .middle } **API reference**

    ---

    Generated reference for every public class and function, with type annotations and
    source links.

    [:octicons-arrow-right-24: Open the reference](api/ome_zarr_container.md)

</div>

## Citing ngio

If ngio contributes to work you publish, please cite it. See
[`CITATION.cff`](https://github.com/BioVisionCenter/ngio/blob/main/CITATION.cff) in the
repository for the current citation metadata.

## Project

ngio is developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html),
University of Zurich, by [@lorenzocerrone](https://github.com/lorenzocerrone) and
[@jluethi](https://github.com/jluethi). It is released under the BSD-3-Clause
[licence](https://github.com/BioVisionCenter/ngio/blob/main/LICENSE), and developed in the
open on [GitHub](https://github.com/BioVisionCenter/ngio) — issues and contributions
welcome.
