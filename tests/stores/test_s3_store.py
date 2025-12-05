# tests/stores/test_s3_store.py

from pathlib import Path

from utils import (
    check_ome_zarr,
    create_sample_ome_zarr,
    derive_image,
    get_s3_mapper,
    random_zarr_path,
)

S3_STORE_SUPPORTED_BACKENDS = ["anndata", "json"]
# CSV and Parquet work locally on S3 store, but not when using moto server for testing.
# To be investigated.


def test_s3_store(moto_s3_server: dict) -> None:
    # create boto3 client pointing at moto server
    endpoint_url, bucket_name = (
        moto_s3_server["endpoint_url"],
        moto_s3_server["bucket_name"],
    )
    store = get_s3_mapper(
        endpoint_url, bucket_name=bucket_name, zarr_path=random_zarr_path()
    )
    ome_zarr = create_sample_ome_zarr(
        store=store, supported_backends=S3_STORE_SUPPORTED_BACKENDS
    )
    check_ome_zarr(ome_zarr, supported_backends=S3_STORE_SUPPORTED_BACKENDS)


def test_s3_store_derive_to_s3_store(moto_s3_server: dict) -> None:
    endpoint_url, bucket_name = (
        moto_s3_server["endpoint_url"],
        moto_s3_server["bucket_name"],
    )
    store = get_s3_mapper(
        endpoint_url,
        bucket_name=bucket_name,
        zarr_path=random_zarr_path(),
    )
    ome_zarr = create_sample_ome_zarr(
        store=store, supported_backends=S3_STORE_SUPPORTED_BACKENDS
    )
    other_store = get_s3_mapper(
        endpoint_url,
        bucket_name=bucket_name,
        zarr_path=random_zarr_path(),
    )
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=S3_STORE_SUPPORTED_BACKENDS)


def test_s3_store_derive_to_local_store(moto_s3_server: dict, tmp_path: Path) -> None:
    endpoint_url, bucket_name = (
        moto_s3_server["endpoint_url"],
        moto_s3_server["bucket_name"],
    )
    store = get_s3_mapper(
        endpoint_url,
        bucket_name=bucket_name,
        zarr_path=random_zarr_path(),
    )
    ome_zarr = create_sample_ome_zarr(
        store=store, supported_backends=S3_STORE_SUPPORTED_BACKENDS
    )
    other_store = tmp_path / "s3_local_store_test" / random_zarr_path()
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=S3_STORE_SUPPORTED_BACKENDS)


def test_s3_store_derive_to_memory_store(moto_s3_server: dict) -> None:
    endpoint_url, bucket_name = (
        moto_s3_server["endpoint_url"],
        moto_s3_server["bucket_name"],
    )
    store = get_s3_mapper(
        endpoint_url,
        bucket_name=bucket_name,
        zarr_path=random_zarr_path(),
    )
    ome_zarr = create_sample_ome_zarr(
        store=store, supported_backends=S3_STORE_SUPPORTED_BACKENDS
    )
    other_store = {}
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=S3_STORE_SUPPORTED_BACKENDS)
