# Quickstart

Ngio is a Python package that provides a simple and intuitive API for reading and writing data to and from OME-Zarr. This guide will walk you through the basics of using `ngio` to read and write data.

## Installation

`ngio` can be installed from PyPI, conda-forge, or from source.

- `ngio` requires Python `>=3.11`

=== "pip"

    The recommended way to install `ngio` is from PyPI using pip:

    ```bash
    pip install ngio
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
    git clone https://github.com/fractal-analytics-platform/ngio.git
    cd ngio
    ```

    2. Install the package:
    ```bash
    pip install .
    ```

### Troubleshooting

Please report installation problems by opening an issue on our [GitHub repository](https://github.com/fractal-analytics-platform/ngio).

## Setup some test data

Let's start by downloading a sample OME-Zarr dataset to work with.

```python exec="true" source="material-block" session="quickstart"
from pathlib import Path
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data")
download_dir = Path(".").absolute() / "data" # markdown-exec: hide
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"
```

## Open an OME-Zarr image

Let's start by opening an OME-Zarr file and inspecting its contents.

```pycon exec="true" source="console" session="quickstart"
>>> from ngio import open_ome_zarr_container
>>> ome_zarr_container = open_ome_zarr_container(image_path)
>>> ome_zarr_container
>>> print(ome_zarr_container) # markdown-exec: hide
```

### What is the OME-Zarr container?

The `OME-Zarr Container` is the core of ngio and the entry point to working with OME-Zarr images. It provides high-level access to the image metadata, images, labels, and tables.

### What is the OME-Zarr container not?

The `OME-Zarr Container` object does not allow the user to interact with the image data directly. For that, we need to use the `Image`, `Label`, and `Table` objects.

## Next steps

To learn how to work with the `OME-Zarr Container` object, but also with the image, label, and table data, check out the following guides:

- [OME-Zarr Container](1_ome_zarr_containers.md): An overview on how to use the OME-Zarr Container object and how to create new images and labels.
- [Images/Labels](2_images.md): To know more on how to work with image data.
- [Tables](3_tables.md): To know more on how to work with table data, and how you can combine tables with image data.
- [Masked Images/Labels](4_masked_images.md): To know more on how to work with masked image data.
- [HCS Plates](5_hcs.md): To know more on how to work with HCS plate data.

Also, checkout our jupyer notebook tutorials for more examples:

- [Image Processing](../tutorials/image_processing.ipynb): Learn how to perform simple image processing operations.
- [Image Segmentation](../tutorials/image_segmentation.ipynb): Learn how to create new labels from images.
- [Feature Extraction](../tutorials/feature_extraction.ipynb): Learn how to extract features from images.
- [HCS Exploration](../tutorials/hcs_exploration.ipynb): Learn how to explore high-content screening data using ngio.
