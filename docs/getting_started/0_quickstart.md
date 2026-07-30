---
description: Install ngio and open your first OME-Zarr container in a few lines of Python.
---

# Quickstart

**Install ngio and open your first OME-Zarr container.**

In a few lines of Python you can open an OME-Zarr store, see what is inside it, and reach
the images, labels and tables it contains.

## Installation

To install `ngio`, use whichever package manager you already work with — it is published on
PyPI and conda-forge, and can also be installed from source.

- `ngio` requires Python `>=3.11`

=== "pip"

    The recommended way to install `ngio` is from PyPI using pip:

    ```bash
    pip install ngio
    ```

=== "uv"

    Inside a uv project:

    ```bash
    uv add ngio
    ```

    Or into an existing environment:

    ```bash
    uv pip install ngio
    ```

=== "pixi"

    ```bash
    pixi add ngio          # from conda-forge
    pixi add --pypi ngio   # from PyPI
    ```

=== "mamba/conda"

    Alternatively, you can install `ngio` using mamba:

    ```bash
    mamba install -c conda-forge ngio
    ```

    or conda:

    ```bash
    conda install -c conda-forge ngio
    ```

=== "Source"

    1. Clone the repository:
    ```bash
    git clone https://github.com/BioVisionCenter/ngio.git
    cd ngio
    ```

    2. Install the package:
    ```bash
    pip install .
    ```

### Troubleshooting

Please report installation problems by opening an issue on the [ngio GitHub repository](https://github.com/BioVisionCenter/ngio).

## Set up test data

Download a sample OME-Zarr dataset to work with.

```python exec="true" source="material-block" session="quickstart"
--8<-- "docs/snippets/getting_started/quickstart.py:setup"
```

## Open an OME-Zarr image

Open an OME-Zarr file and inspect its contents.

```python exec="true" source="material-block" session="quickstart"
--8<-- "docs/snippets/getting_started/quickstart.py:open_container"
```

The pixels are one call away — here is the DAPI channel of that container, read from a
lower pyramid level:

```python exec="true" session="quickstart"
--8<-- "docs/snippets/getting_started/quickstart.py:plot_helpers"
```
```python exec="true" html="1" session="quickstart"
--8<-- "docs/snippets/getting_started/quickstart.py:plot_quickstart_image"
```

### What is the OME-Zarr container?

The OME-Zarr container is the core of ngio and the entry point to working with OME-Zarr images. It provides high-level access to the image metadata, images, labels, and tables. The [next section](1_ome_zarr_containers.md) goes into more detail: inspecting and editing metadata, opening remote stores, and deriving new images and labels.

### What is the OME-Zarr container not?

The OME-Zarr container does not give you access to the image data directly. For that, use the `Image`, `Label`, and `Table` objects.

## Next steps

- [OME-Zarr containers](1_ome_zarr_containers.md) — inspect and modify metadata, and create new images and labels.
- [Images and labels](2_images.md) — read and write pixel data.
- [Tables](3_tables.md) — ROIs, features and measurements stored alongside the image.
- [Masked images and labels](4_masked_images.md) — work object-by-object using a segmentation.
- [HCS plates](5_hcs.md) — scale up from a single image to a whole plate.

For worked end-to-end examples, see the tutorials:

- [Image processing](../tutorials/image_processing.md) — apply a processing step across an image.
- [Image segmentation](../tutorials/image_segmentation.md) — create new labels from images.
- [Feature extraction](../tutorials/feature_extraction.md) — measure objects and store the results.
- [HCS exploration](../tutorials/hcs_exploration.md) — navigate high-content screening data.
