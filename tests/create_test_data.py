from pathlib import Path

from ngio import NgffVersions, create_empty_ome_zarr

DATA_DIR = Path("tests/data")

IMAGE_SPECS = [
    {
        "name": "test_image_tcyx.zarr",
        "shape": (4, 3, 64, 64),
        "axes": "tcyx",
    },
    {
        "name": "test_image_tczyx.zarr",
        "shape": (4, 2, 10, 64, 64),
        "axes": "tczyx",
    },
    {
        "name": "test_image_zyx.zarr",
        "shape": (10, 64, 64),
        "axes": "zyx",
    },
    {
        "name": "test_image_c1yx.zarr",
        "shape": (2, 1, 64, 64),
        "axes": "czyx",
    },
    {
        "name": "test_image_tyx.zarr",
        "shape": (4, 64, 64),
        "axes": "tyx",
    },
    {
        "name": "test_image_tzyx.zarr",
        "shape": (4, 10, 64, 64),
        "axes": "tzyx",
    },
    {
        "name": "test_image_cyx.zarr",
        "shape": (2, 64, 64),
        "axes": "cyx",
    },
    {
        "name": "test_image_yx.zarr",
        "shape": (64, 64),
        "axes": "yx",
    },
    {
        "name": "test_image_czyx.zarr",
        "shape": (2, 10, 64, 64),
        "axes": "czyx",
    },
]


def create_test_images_dataset(version: NgffVersions) -> None:
    version_str = "".join(version.split("."))
    base_dir = DATA_DIR / f"v{version_str}" / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    for spec in IMAGE_SPECS:
        image_path = base_dir / spec["name"]
        ome_zarr = create_empty_ome_zarr(
            store=image_path,
            xy_pixelsize=0.5,
            shape=spec["shape"],
            axes_names=spec["axes"],
            ngff_version=version,
            levels=2,
            overwrite=True,
        )
        ome_zarr.derive_label("label")
        well_roi_table = ome_zarr.build_image_roi_table()
        ome_zarr.add_table(name="well_ROI_table", table=well_roi_table, backend="csv")


if __name__ == "__main__":
    for version in ["0.4", "0.5"]:
        create_test_images_dataset(version)  # type: ignore
