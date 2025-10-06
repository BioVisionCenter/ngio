"""Download testing OME-Zarr datasets."""

import shutil
from pathlib import Path
from typing import Literal

import pooch

from ngio.utils._errors import NgioValueError


class UnzipAndRename(pooch.Unzip):
    """Unzip and rename the extracted directory."""

    def __init__(
        self,
        extract_dir: str = "",
        out_name: str = "ome-zarr.zarr",
        re_unzip: bool = True,
        **kwargs,
    ):
        super().__init__(extract_dir=extract_dir, **kwargs)
        self.out_name = out_name
        self.re_unzip = re_unzip

    def output_file(self) -> Path:
        """Return the output file path."""
        if self.extract_dir is None:
            raise NgioValueError("extract_dir must be set for UnzipAndRename.")

        return Path(self.extract_dir) / self.out_name

    def _extract_file(self, fname, extract_dir):
        """Extract the file and rename it."""
        output_path = self.output_file()
        if output_path.exists() and not self.re_unzip:
            # Nothing to do, the file already exists and we are not re-unzipping
            return None

        tmp_dir = Path(extract_dir) / "tmp"
        # If tmp_dir exists, remove it
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        super()._extract_file(fname, tmp_dir)

        list_extracted_dirs = tmp_dir.iterdir()
        # Keep only if ends with .zarr
        list_extracted_dirs = filter(
            lambda x: x.name.endswith(".zarr"),
            list_extracted_dirs,
        )
        list_extracted_dirs = list(list_extracted_dirs)
        if len(list_extracted_dirs) != 1:
            raise NgioValueError(
                "Expected one directory to be extracted, "
                f"got {len(list_extracted_dirs)}."
            )

        extracted_dir = list_extracted_dirs[0]
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)

        extracted_dir.rename(output_path)
        # Clean up the temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


_ome_zarr_zoo = {
    "CardiomyocyteTiny": {
        "url": "https://zenodo.org/records/13305156/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip",
        "known_hash": "md5:efc21fe8d4ea3abab76226d8c166452c",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1-tiny.zarr.zip",
        "description": "Tiny cardiomyocyte dataset 3D (32MB).",
    },
    "CardiomyocyteTinyMip": {
        "url": "https://zenodo.org/records/13305156/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "known_hash": "md5:51809479777cafbe9ac0f9fa5636aa95",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1-tiny-mip.zarr.zip",
        "description": "Tiny cardiomyocyte dataset 2D MIP (16.4MB).",
    },
    "CardiomyocyteSmall": {
        "url": "https://zenodo.org/records/13305316/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip",
        "known_hash": "md5:d5752ed4b72a9092a0290b3c04c0b9c2",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1-small.zarr.zip",
        "description": "Small cardiomyocyte dataset 3D (750MB).",
    },
    "CardiomyocyteSmallMip": {
        "url": "https://zenodo.org/records/13305316/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "known_hash": "md5:3ed3ea898e0ed42d397da2e1dbe40750",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1-small-mip.zarr.zip",
        "description": "Small cardiomyocyte dataset 2D MIP (106MB).",
    },
    "CardiomyocyteMediumMip": {
        "url": "https://zenodo.org/records/14826000/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "known_hash": "md5:3f932bbf7fc0577f58b97471707816a1",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "description": "Medium cardiomyocyte dataset 2D MIP (30GB).",
    },
}

AVAILABLE_DATASETS = Literal[
    "CardiomyocyteTiny",
    "CardiomyocyteTinyMip",
    "CardiomyocyteSmall",
    "CardiomyocyteSmallMip",
    "CardiomyocyteMediumMip",
]


def list_ome_zarr_datasets() -> list[str]:
    """List available OME-Zarr datasets."""
    return list(_ome_zarr_zoo.keys())


def print_datasets_infos() -> None:
    for dataset_name, dataset_info in _ome_zarr_zoo.items():
        print(f"{dataset_name} - Description: {dataset_info['description']}")


def download_ome_zarr_dataset(
    dataset_name: AVAILABLE_DATASETS | str,
    download_dir: str | Path = "data",
    re_unzip: bool = True,
    progressbar: bool = False,
) -> Path:
    """Download an OME-Zarr dataset.

    To list available datasets, use `list_ome_zarr_datasets`.

    Args:
        dataset_name (str): The dataset name.
        download_dir (str): The download directory. Defaults to "data".
        re_unzip (bool): If True, it will unzip the dataset even if it already exists.
        progressbar (bool): If True, show a progress bar during download.
    """
    if dataset_name not in _ome_zarr_zoo:
        raise NgioValueError(f"Dataset {dataset_name} not found in the OME-Zarr zoo.")
    zenodo_infos = _ome_zarr_zoo[dataset_name]

    fname = zenodo_infos["fname"]
    zarrname = fname.replace(".zip", "")

    processor = UnzipAndRename(
        extract_dir="",
        out_name=zarrname,
        re_unzip=re_unzip,
    )

    pooch.retrieve(
        url=zenodo_infos["url"],
        known_hash=zenodo_infos["known_hash"],
        fname=fname,
        path=download_dir,
        processor=processor,
        progressbar=progressbar,
    )
    return processor.output_file()
