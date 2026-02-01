from pathlib import Path

from utils import (
    check_ome_zarr,
    create_sample_ome_zarr,
    derive_image,
    get_s3_mapper,
    random_zarr_path,
)

MEMORY_STORE_SUPPORTED_BACKENDS = ["anndata", "json", "csv", "parquet"]


def test_memory_store() -> None:
    store_path = {}
    ome_zarr = create_sample_ome_zarr(
        store=store_path, supported_backends=MEMORY_STORE_SUPPORTED_BACKENDS
    )
    check_ome_zarr(ome_zarr, supported_backends=MEMORY_STORE_SUPPORTED_BACKENDS)


def test_memory_store_derive_to_memory_store() -> None:
    store_path = {}
    ome_zarr = create_sample_ome_zarr(
        store=store_path, supported_backends=MEMORY_STORE_SUPPORTED_BACKENDS
    )
    other_store = {}
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=["anndata", "json"])


def test_memory_store_derive_to_local_store(tmp_path: Path) -> None:
    store_path = {}
    ome_zarr = create_sample_ome_zarr(
        store=store_path, supported_backends=MEMORY_STORE_SUPPORTED_BACKENDS
    )
    other_store = tmp_path / "local_store_test" / "derived_test.zarr"
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=["anndata", "json"])


def test_memory_store_derive_to_s3_store(moto_s3_server: dict) -> None:
    store_path = {}
    ome_zarr = create_sample_ome_zarr(
        store=store_path, supported_backends=MEMORY_STORE_SUPPORTED_BACKENDS
    )
    # create boto3 client pointing at moto server
    other_store = get_s3_mapper(
        base_url=moto_s3_server["endpoint_url"],
        bucket_name=moto_s3_server["bucket_name"],
        zarr_path=random_zarr_path(),
    )
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=["anndata", "json"])
