from pathlib import Path

from utils import (
    check_ome_zarr,
    create_sample_ome_zarr,
    derive_image,
    get_http_mapper,
    get_s3_mapper,
    random_zarr_path,
)

from ngio import open_ome_zarr_container

HTTP_STORE_SUPPORTED_BACKENDS = ["anndata", "json", "csv", "parquet"]


def test_http_store(http_static_server: dict) -> None:
    # create boto3 client pointing at moto server
    url, root = http_static_server["url"], http_static_server["root"]

    zarr_path = random_zarr_path()
    local_store = root / zarr_path
    _ = create_sample_ome_zarr(
        store=local_store, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS
    )
    http_mapper = get_http_mapper(url, zarr_path)
    ome_zarr = open_ome_zarr_container(store=http_mapper)
    check_ome_zarr(ome_zarr, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS)


def test_http_store_derive_to_s3_store(
    http_static_server: dict, moto_s3_server: dict
) -> None:
    url, root = http_static_server["url"], http_static_server["root"]
    zarr_path = random_zarr_path()
    local_store = root / zarr_path
    _ = create_sample_ome_zarr(
        store=local_store, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS
    )
    http_mapper = get_http_mapper(url, zarr_path)
    ome_zarr = open_ome_zarr_container(store=http_mapper)
    other_store = get_s3_mapper(
        base_url=moto_s3_server["endpoint_url"],
        bucket_name=moto_s3_server["bucket_name"],
        zarr_path=random_zarr_path(),
    )
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS)


def test_http_store_derive_to_local_store(
    http_static_server: dict, tmp_path: Path
) -> None:
    url, root = http_static_server["url"], http_static_server["root"]
    zarr_path = random_zarr_path()
    local_store = root / zarr_path
    _ = create_sample_ome_zarr(
        store=local_store, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS
    )
    http_mapper = get_http_mapper(url, zarr_path)
    ome_zarr = open_ome_zarr_container(store=http_mapper)

    other_store = tmp_path / "http_local_store_test" / random_zarr_path()
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS)


def test_http_store_derive_to_memory_store(http_static_server: dict) -> None:
    url, root = http_static_server["url"], http_static_server["root"]
    zarr_path = random_zarr_path()
    local_store = root / zarr_path
    _ = create_sample_ome_zarr(
        store=local_store, supported_backends=HTTP_STORE_SUPPORTED_BACKENDS
    )
    http_mapper = get_http_mapper(url, zarr_path)
    ome_zarr = open_ome_zarr_container(store=http_mapper)

    other_store = {}
    derived_ome_zarr = derive_image(ome_zarr, other_store=other_store)
    check_ome_zarr(derived_ome_zarr, supported_backends=["anndata", "json"])
