from uuid import uuid4

import fsspec.implementations.http
import numpy as np
import pandas as pd
import s3fs

from ngio import OmeZarrContainer, create_empty_ome_zarr
from ngio.tables import FeatureTable

TEST_IMAGE = np.ones((3, 5, 64, 64), dtype=np.uint16)
TEST_LABEL = np.zeros((5, 64, 64), dtype=np.uint16)
TEST_LABEL[0, 10:30, 10:30] = 1
TEST_LABEL[0, 35:55, 35:55] = 2
TEST_LABEL[0, 20:40, 40:60] = 3

TEST_TABLE = pd.DataFrame(
    {
        "label": [1, 2, 3],
        "area": [150, 200, 250],
        "mean_intensity": [120.5, 130.0, 140.5],
    }
)
TEST_TABLE.set_index("label", inplace=True)


def set_to_image(ome_zarr: OmeZarrContainer) -> OmeZarrContainer:
    """Create and return an empty OME-Zarr container in the given store."""
    image = ome_zarr.get_image(path="1")
    image.set_array(patch=TEST_IMAGE)
    image.consolidate()
    return ome_zarr


def check_image_data(
    ome_zarr: OmeZarrContainer,
) -> None:
    """Retrieve the array data from the first image at the specified pyramid level."""
    image = ome_zarr.get_image()
    array_data = image.get_as_numpy()
    np.testing.assert_array_equal(array_data, TEST_IMAGE)


def derive_label(
    ome_zarr: OmeZarrContainer,
) -> OmeZarrContainer:
    """Derive a label array from the first image in the OME-Zarr container."""
    label_array = ome_zarr.derive_label(name="labels")
    label_array.set_array(patch=TEST_LABEL)
    label_array.consolidate()
    return ome_zarr


def check_label_data(
    ome_zarr: OmeZarrContainer,
) -> None:
    """Retrieve the label data from the specified label array."""
    label_array = ome_zarr.get_label(name="labels")
    label_data = label_array.get_as_numpy()
    np.testing.assert_array_equal(label_data, TEST_LABEL)


def add_table_to_ome_zarr(
    ome_zarr: OmeZarrContainer,
    backend: str = "anndata",
) -> OmeZarrContainer:
    """Add a table to the OME-Zarr container."""
    test_feature_table = FeatureTable(table_data=TEST_TABLE, reference_label="labels")
    name = f"test_table_{backend}"
    ome_zarr.add_table(
        name=name,
        table=test_feature_table,
        backend=backend,
    )
    return ome_zarr


def check_table_data(
    ome_zarr: OmeZarrContainer,
    table_name: str,
) -> None:
    """Retrieve the table data from the specified table in the OME-Zarr container."""
    table = ome_zarr.get_table(name=table_name)
    dataframe = table.dataframe
    pd.testing.assert_frame_equal(
        dataframe.sort_index(axis=1), TEST_TABLE.sort_index(axis=1)
    )


def derive_image(
    ome_zarr: OmeZarrContainer,
    other_store,
    copy_labels: bool = True,
    copy_tables: bool = True,
) -> OmeZarrContainer:
    """Derive a new image from the first image in the OME-Zarr container."""
    derived_ome_zarr = ome_zarr.derive_image(
        store=other_store,
        copy_labels=copy_labels,
        copy_tables=copy_tables,
    )
    return derived_ome_zarr


def create_sample_ome_zarr(
    store, supported_backends: list[str] | None = None
) -> OmeZarrContainer:
    """Create a sample OME-Zarr structure in the given store for testing."""
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(3, 5, 64, 64),
        pixelsize=(0.65, 0.65),
        channels_meta=["Channel 1", "Channel 2", "Channel 3"],
        levels=3,
        axes_names=["c", "z", "y", "x"],
    )
    ome_zarr = set_to_image(ome_zarr)
    check_image_data(ome_zarr)
    ome_zarr = derive_label(ome_zarr)
    if supported_backends is None:
        supported_backends = ["anndata", "json", "csv", "parquet"]

    for backend in supported_backends:
        ome_zarr = add_table_to_ome_zarr(ome_zarr, backend=backend)
    return ome_zarr


def check_ome_zarr(
    ome_zarr: OmeZarrContainer,
    check_tables: bool = True,
    check_labels: bool = True,
    supported_backends: list[str] | None = None,
) -> None:
    """Check that all tables in the OME-Zarr container match the test table."""
    if check_labels:
        check_label_data(ome_zarr)
    if check_tables:
        if supported_backends is None:
            supported_backends = ["anndata", "json", "csv", "parquet"]

        table_list = ome_zarr.list_tables()
        assert len(table_list) >= len(supported_backends)
        for table_name in table_list:
            backend = table_name.split("_")[-1]
            if backend in supported_backends:
                check_table_data(ome_zarr, table_name=table_name)


def get_s3_mapper(base_url: str, bucket_name: str, zarr_path: str):
    s3_fs = s3fs.S3FileSystem(
        key="test",
        secret="test",
        client_kwargs={
            "endpoint_url": base_url,
            "region_name": "us-east-1",
        },
    )
    return s3_fs.get_mapper(f"{bucket_name}/{zarr_path}")


def get_http_mapper(base_url: str, zarr_path: str):
    http_fs = fsspec.implementations.http.HTTPFileSystem()
    return http_fs.get_mapper(f"{base_url}/{zarr_path}")


def random_zarr_path() -> str:
    """Generate a random zarr path for testing."""
    return f"test_zarr_{uuid4()}.zarr"
