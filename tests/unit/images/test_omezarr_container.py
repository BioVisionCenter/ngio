from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from ngio import (
    create_empty_ome_zarr,
    create_synthetic_ome_zarr,
    open_ome_zarr_container,
)
from ngio.images._image import ChannelSelectionModel
from ngio.io_pipes import TransformContext
from ngio.ome_zarr_meta import ChannelsMeta
from ngio.tables import GenericTable
from ngio.utils import NgioValueError, fractal_fsspec_store


class IdentityTransform:
    def on_get(
        self, array: np.ndarray | da.Array, ctx: TransformContext
    ) -> np.ndarray | da.Array:
        """Apply the transformation after reading."""
        return array

    def on_set(
        self, array: np.ndarray | da.Array, ctx: TransformContext
    ) -> np.ndarray | da.Array:
        """Apply the inverse transformation before writing."""
        return array


def test_open_ome_zarr_container(
    images_all_versions_readonly: dict[str, Path], zarr_name: str
):
    path = images_all_versions_readonly[zarr_name]
    ome_zarr = open_ome_zarr_container(path)

    whole_image_roi = ome_zarr.build_image_roi_table().get("image")
    image = ome_zarr.get_image()
    assert isinstance(image.__repr__(), str)
    assert image.get_roi(whole_image_roi).shape == image.shape

    label = ome_zarr.get_label("label")
    assert isinstance(label.__repr__(), str)
    roi = image.build_image_roi_table().get("image")
    image.get_roi(roi)
    label.get_roi(roi)


def test_ome_zarr_tables(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03" / "0"
    ome_zarr = open_ome_zarr_container(cardiomyocyte_tiny_path)

    assert isinstance(ome_zarr.__repr__(), str)
    _ = ome_zarr.images_container

    assert not ome_zarr.is_2d
    assert ome_zarr.is_3d
    assert not ome_zarr.is_time_series
    assert not ome_zarr.is_multi_channels
    assert not ome_zarr.is_2d_time_series
    assert not ome_zarr.is_3d_time_series
    assert ome_zarr.channel_labels == ["DAPI"]
    assert ome_zarr.wavelength_ids == ["A01_C01"]
    assert ome_zarr.get_channel_idx("DAPI") == 0
    assert ome_zarr.get_channel_idx(wavelength_id="A01_C01") == 0
    assert ome_zarr.num_channels == 1
    ome_zarr.set_space_unit("micrometer")

    assert ome_zarr.list_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        ome_zarr.list_tables()
    )
    assert ome_zarr.list_roi_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        ome_zarr.list_roi_tables()
    )

    assert ome_zarr.list_labels() == []

    fov_roi = ome_zarr.get_table("FOV_ROI_table")
    fov_roi = ome_zarr.get_roi_table("FOV_ROI_table")
    assert len(fov_roi.rois()) == 2
    roi_table_1 = ome_zarr.get_roi_table("well_ROI_table")
    assert len(roi_table_1.rois()) == 1

    new_well_roi_table = ome_zarr.build_image_roi_table()
    ome_zarr.add_table("new_well_ROI_table", new_well_roi_table)

    assert ome_zarr.list_tables() == [
        "FOV_ROI_table",
        "well_ROI_table",
        "new_well_ROI_table",
    ]


@pytest.mark.parametrize("array_mode", ["numpy", "dask"])
def test_create_ome_zarr_container(tmp_path: Path, array_mode: str):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(10, 20, 30),
        chunks=(1, 20, 30),
        pixelsize=0.5,
        levels=3,
        dtype="uint8",
    )

    assert isinstance(ome_zarr.__repr__(), str)
    assert ome_zarr.levels == 3
    assert ome_zarr.level_paths == ["0", "1", "2"]
    assert ome_zarr.is_3d
    assert not ome_zarr.is_time_series
    assert not ome_zarr.is_multi_channels
    assert not ome_zarr.is_2d_time_series
    assert not ome_zarr.is_3d_time_series
    assert ome_zarr.space_unit == "micrometer"
    assert ome_zarr.time_unit is None

    ome_zarr.set_space_unit("yoctometer")
    ome_zarr.set_time_unit("yoctosecond")
    assert ome_zarr.space_unit == "yoctometer"
    assert ome_zarr.time_unit is None

    image = ome_zarr.get_image()

    assert image.shape == (10, 20, 30)
    assert image.dtype == "uint8"
    assert image.chunks == (1, 20, 30)
    assert image.pixel_size.x == 0.5
    assert image.meta.get_highest_resolution_dataset().path == "0"
    assert image.meta.get_lowest_resolution_dataset().path == "2"

    array = image.get_as_numpy(transforms=[IdentityTransform()])
    assert isinstance(array, np.ndarray)
    assert array.shape == (10, 20, 30)

    array_dask = image.get_as_dask(transforms=[IdentityTransform()])
    assert array_dask.shape == (10, 20, 30)

    array = image.get_array(
        x=slice(None),
        axes_order=["c", "z", "y", "x"],
        mode=array_mode,  # type: ignore
    )

    array = array + 1  # type: ignore

    image.set_array(
        array,
        x=slice(None),
        axes_order=["c", "z", "y", "x"],
        transforms=[IdentityTransform()],
    )
    image.consolidate(mode=array_mode)  # type: ignore

    # Omero-meta
    ome_zarr.set_channel_meta(
        channel_meta=ChannelsMeta.default_init(
            labels=["channel_x"], wavelength_id=["500"]
        )
    )
    image = ome_zarr.get_image()
    assert image.channel_labels == ["channel_x"]
    assert image.wavelength_ids == ["500"]

    ome_zarr.set_channel_labels(labels=["channel_y"])
    image = ome_zarr.get_image()
    assert image.channel_labels == ["channel_y"]
    assert image.wavelength_ids == ["500"]
    ome_zarr.set_name("test_image")
    assert ome_zarr.meta.name == "test_image"

    ome_zarr.set_channel_colors(colors=["00FF00"])
    channels_meta = ome_zarr.meta.channels_meta
    assert channels_meta is not None
    assert channels_meta.channels[0].channel_visualisation.color == "00FF00"

    ome_zarr.set_channel_windows_with_percentiles()
    ome_zarr.set_channel_windows(starts_ends=[(10, 200)], min_max=[(0, 255)])
    channels_meta = ome_zarr.meta.channels_meta
    assert channels_meta is not None
    assert len(channels_meta.channels) == 1
    assert channels_meta.channels[0].label == "channel_y"
    assert channels_meta.channels[0].wavelength_id == "500"
    assert channels_meta.channels[0].channel_visualisation.start == 10
    assert channels_meta.channels[0].channel_visualisation.end == 200
    assert channels_meta.channels[0].channel_visualisation.min == 0
    assert channels_meta.channels[0].channel_visualisation.max == 255

    image = ome_zarr.get_image(path="2")
    assert np.mean(image.get_array()) == 1  # type: ignore

    new_ome_zarr = ome_zarr.derive_image(tmp_path / "derived2.zarr", ref_path="2")

    assert new_ome_zarr.levels == 3
    new_image = new_ome_zarr.get_image()
    assert new_image.shape == image.shape

    new_label = new_ome_zarr.derive_label("new_label")
    assert new_label.shape == image.shape
    assert new_label.meta.axes_handler.axes_names == ("z", "y", "x")

    assert new_ome_zarr.list_labels() == ["new_label"]
    assert new_ome_zarr.list_tables() == []
    assert new_ome_zarr.list_roi_tables() == []

    # Test masked image instantiation
    masked_image = new_ome_zarr.get_masked_image(masking_label_name="new_label")
    assert masked_image.shape == image.shape
    masked_label = new_ome_zarr.get_masked_label(
        label_name="new_label", masking_label_name="new_label"
    )
    assert masked_label.shape == image.shape


@pytest.mark.network
def test_remote_ome_zarr_container():
    url = (
        "https://raw.githubusercontent.com/"
        "fractal-analytics-platform/fractal-ome-zarr-examples/"
        "refs/heads/main/v04/"
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
    )

    store = fractal_fsspec_store(url)
    ome_zarr = open_ome_zarr_container(store)

    assert ome_zarr.list_labels() == ["nuclei"]

    _ = ome_zarr.get_label("nuclei", path="0")
    _ = ome_zarr.get_table("well_ROI_table").dataframe


def test_get_and_squeeze(tmp_path: Path):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(1, 20, 30),
        pixelsize=0.5,
        levels=1,
        axes_names=["c", "y", "x"],
        dtype="uint8",
    )
    image = ome_zarr.get_image()
    assert image.shape == (1, 20, 30)
    assert image.get_array(axes_order=["c", "y", "x"]).shape == (1, 20, 30)
    image.set_array(
        np.ones((1, 20, 30), dtype="uint8"),
        axes_order=["c", "y", "x"],
    )
    assert image.get_array(axes_order=["y", "x"]).shape == (20, 30)
    image.set_array(
        np.ones((20, 30), dtype="uint8"),
        axes_order=["y", "x"],
    )
    assert image.get_array(axes_order=["y", "x"]).shape == (20, 30)
    image.set_array(
        np.ones((20, 30), dtype="uint8"),
        axes_order=["y", "x"],
    )
    assert image.get_array(axes_order=["x"], y=0, c=0).shape == (30,)
    image.set_array(
        np.ones((30,), dtype="uint8"),
        axes_order=["x"],
        y=0,
        c=0,
    )

    assert image.get_array(channel_selection=0, axes_order=["c", "y", "x"]).shape == (
        1,
        20,
        30,
    )
    assert image.get_array(channel_selection=(0,), axes_order=["y", "x"]).shape == (
        20,
        30,
    )
    assert image.get_array(channel_selection=(0,), axes_order=None).shape == (1, 20, 30)
    assert image.get_array(channel_selection=0, axes_order=None).shape == (20, 30)

    # Reordering axes and adding a virtual axis
    assert image.get_array(
        channel_selection=0, axes_order=["c", "x", "y", "virtual"]
    ).shape == (
        1,
        30,
        20,
        1,
    )

    # Test channel_labels
    image.get_as_numpy(channel_selection="channel_0")
    image.get_as_dask(channel_selection="channel_0")
    image.get_as_dask(
        channel_selection=ChannelSelectionModel(identifier="channel_0", mode="label")
    )
    image.get_as_dask(
        channel_selection=ChannelSelectionModel(identifier="0", mode="index")
    )


def test_derive_image_and_labels(tmp_path: Path):
    # Testing for #116
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(3, 20, 30),
        pixelsize=0.5,
        levels=1,
        axes_names=["c", "y", "x"],
        dtype="uint8",
    )
    derived_ome_zarr = ome_zarr.derive_image(tmp_path / "derived.zarr")
    _ = derived_ome_zarr.derive_label("derived_label")


def test_add_table_preserves_backend(tmp_path: Path):
    # Testing for #207
    ome_zarr = create_synthetic_ome_zarr(
        tmp_path / "src.zarr",
        shape=(3, 20, 30),
        levels=1,
        axes_names=["c", "y", "x"],
    )
    table = GenericTable(table_data=pd.DataFrame({"a": [1, 2, 3]}))
    ome_zarr.add_table("parquet_table", table, backend="parquet")

    dst_ome_zarr = create_empty_ome_zarr(
        tmp_path / "dst.zarr", shape=(3, 20, 30), pixelsize=0.5
    )
    loaded_table = ome_zarr.get_table("parquet_table")
    dst_ome_zarr.add_table("parquet_table", loaded_table)
    assert dst_ome_zarr.get_table("parquet_table").backend_name == "parquet"


def test_derive_copy_labels_and_tables(tmp_path: Path):
    # Testing for #116
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store,
        shape=(3, 20, 30),
        levels=1,
        axes_names=["c", "y", "x"],
    )
    derived_ome_zarr = ome_zarr.derive_image(
        tmp_path / "derived.zarr", copy_labels=True, copy_tables=True
    )
    assert ome_zarr.list_labels() == derived_ome_zarr.list_labels()
    assert ome_zarr.list_tables() == derived_ome_zarr.list_tables()


def test_delete_label_and_table(tmp_path: Path):
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store,
        shape=(3, 20, 30),
        levels=1,
        axes_names=["c", "y", "x"],
    )
    ome_zarr.derive_label("label_to_delete")
    assert "label_to_delete" in ome_zarr.list_labels()
    ome_zarr.delete_label("label_to_delete")
    assert "label_to_delete" not in ome_zarr.list_labels()
    ome_zarr.delete_label("label_to_delete", missing_ok=True)
    with pytest.raises(NgioValueError):
        ome_zarr.delete_label("label_to_delete", missing_ok=False)

    new_table = ome_zarr.build_image_roi_table()
    ome_zarr.add_table("table_to_delete", new_table)
    assert "table_to_delete" in ome_zarr.list_tables()
    ome_zarr.delete_table("table_to_delete")
    assert "table_to_delete" not in ome_zarr.list_tables()
    ome_zarr.delete_table("table_to_delete", missing_ok=True)
    with pytest.raises(NgioValueError):
        ome_zarr.delete_table("table_to_delete", missing_ok=False)

    ome_zarr = create_empty_ome_zarr(
        store, shape=(3, 20, 30), pixelsize=0.5, overwrite=True
    )
    with pytest.raises(NgioValueError):
        ome_zarr.delete_label("non_existing_label")
    ome_zarr.delete_label("non_existing_label", missing_ok=True)
    with pytest.raises(NgioValueError):
        ome_zarr.delete_table("non_existing_table")
    ome_zarr.delete_table("non_existing_table", missing_ok=True)


def test_rename_axes():
    ome_zarr = create_empty_ome_zarr(
        store={},
        shape=(3, 127, 128),
        pixelsize=1.0,
        translation=(0.0, 1.0, 1.0),
        ngff_version="0.5",
        axes_names=["c", "y", "XX"],
        overwrite=True,
    )

    ome_zarr.derive_label("label_1")
    assert ome_zarr.get_image().axes == ("c", "y", "XX")
    label = ome_zarr.get_label("label_1")
    assert label.axes == ("y", "XX")
    ome_zarr.set_axes_names(axes_names=["c", "y", "x"])
    assert ome_zarr.get_image().axes == ("c", "y", "x")


def test_uncached_container_sees_tables_added_by_another_handle(tmp_path: Path):
    """The "no /tables" answer may only be remembered under `cache=True`."""
    store = tmp_path / "ome_zarr.zarr"
    create_empty_ome_zarr(store, shape=(3, 20, 30), pixelsize=0.5)

    reader = open_ome_zarr_container(store, cache=False)
    assert reader.list_tables() == []

    writer = open_ome_zarr_container(store)
    writer.add_table("t", GenericTable(table_data=pd.DataFrame({"a": [1]})))

    assert reader.list_tables() == ["t"]


def test_cached_container_holds_the_no_tables_answer(tmp_path: Path):
    store = tmp_path / "ome_zarr.zarr"
    create_empty_ome_zarr(store, shape=(3, 20, 30), pixelsize=0.5)

    reader = open_ome_zarr_container(store, cache=True)
    assert reader.list_tables() == []

    writer = open_ome_zarr_container(store)
    writer.add_table("t", GenericTable(table_data=pd.DataFrame({"a": [1]})))

    assert reader.list_tables() == []
    reader.refresh()
    assert reader.list_tables() == ["t"]
