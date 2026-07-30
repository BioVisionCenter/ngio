"""Regression tests for the API decisions frozen by ngio 1.0."""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from ngio import (
    OmeZarrContainer,
    create_empty_ome_zarr,
    open_image,
    open_label,
)
from ngio.utils import NgioDeprecationWarning


@pytest.fixture
def float_container(tmp_path: Path) -> OmeZarrContainer:
    return create_empty_ome_zarr(
        store=tmp_path / "float.zarr",
        shape=(32, 32),
        pixelsize=0.5,
        dtype="float32",
    )


def test_derive_image_inherits_dtype(float_container, tmp_path):
    """`derive_image` used to hardcode uint16 while documenting inheritance."""
    derived = float_container.derive_image(store=tmp_path / "derived.zarr")
    assert derived.get_image().dtype == "float32"


def test_derive_image_dtype_can_still_be_overridden(float_container, tmp_path):
    derived = float_container.derive_image(
        store=tmp_path / "derived_u8.zarr", dtype="uint8"
    )
    assert derived.get_image().dtype == "uint8"


def test_derive_image_inherits_compressors_and_separator(float_container, tmp_path):
    source = float_container.get_image().zarr_array
    derived = float_container.derive_image(store=tmp_path / "derived_c.zarr")
    assert derived.get_image().zarr_array.compressors == source.compressors


def test_derive_label_from_image_is_uint32(float_container):
    """Deriving a label from an image keeps its uint32 special case."""
    label = float_container.derive_label(name="seg")
    assert label.dtype == "uint32"


def test_derive_label_from_label_inherits(float_container):
    float_container.derive_label(name="base", dtype="uint16")
    base = float_container.get_label("base")
    derived = float_container.derive_label(name="copy", ref_image=base)
    assert derived.dtype == "uint16"


def test_list_roi_tables_returns_empty_on_container(float_container):
    """Used to raise NgioValidationError while list_tables returned []."""
    assert float_container.list_tables() == []
    assert float_container.list_roi_tables() == []


def test_list_roi_tables_returns_empty_on_plate(tmp_path):
    from ngio import create_empty_plate

    plate = create_empty_plate(store=tmp_path / "plate.zarr", name="p")
    assert plate.list_tables() == []
    assert plate.list_roi_tables() == []


def test_open_image_strict_default_matches_get_image(tmp_path):
    """`open_image` defaulted to strict=True while `get_image` defaulted to False."""
    import inspect

    from ngio.images import ImagesContainer

    assert (
        inspect.signature(open_image).parameters["strict"].default
        == inspect.signature(ImagesContainer.get).parameters["strict"].default
    )
    assert inspect.signature(open_image).parameters["strict"].default is False
    assert inspect.signature(open_label).parameters["strict"].default is False


def test_validate_arrays_is_forwarded(tmp_path, monkeypatch):
    """OmeZarrContainer used to drop the flag before its ImagesContainer."""
    from ngio.images import _image as image_module

    store = tmp_path / "v.zarr"
    create_empty_ome_zarr(store=store, shape=(32, 32), pixelsize=0.5, levels=3)

    opened = []
    original = image_module.ImagesContainer.get

    def spy(self, *args, **kwargs):
        opened.append(kwargs.get("path"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(image_module.ImagesContainer, "get", spy)

    from ngio import open_ome_zarr_container

    open_ome_zarr_container(store, validate_arrays=False)
    assert opened == []

    opened.clear()
    open_ome_zarr_container(store, validate_arrays=True)
    assert len(opened) == 3


def test_validate_paths_alias_warns(tmp_path):
    from ngio.images import ImagesContainer
    from ngio.utils import ZarrGroupHandler

    store = tmp_path / "alias.zarr"
    create_empty_ome_zarr(store=store, shape=(32, 32), pixelsize=0.5)
    handler = ZarrGroupHandler(store=store, mode="r+")
    with pytest.warns(NgioDeprecationWarning, match="validate_paths"):
        ImagesContainer(handler, axes_setup=None, validate_paths=False)


def test_set_axes_unit_alias_warns(float_container):
    with pytest.warns(NgioDeprecationWarning, match="set_axes_units"):
        float_container.get_image().set_axes_unit(space_unit="millimeter")
    assert float_container.get_image().space_unit == "millimeter"


def test_levels_paths_alias_warns():
    from ngio.common._pyramid import ImagePyramidBuilder

    with pytest.warns(NgioDeprecationWarning, match="level_paths"):
        builder = ImagePyramidBuilder.from_shapes(
            shapes=[(32, 32), (16, 16)],
            base_scale=(1.0, 1.0),
            axes=("y", "x"),
            levels_paths=["a", "b"],
        )
    assert [level.path for level in builder.levels] == ["a", "b"]


def test_conctatenate_tables_typo_alias_warns(tmp_path):
    import pandas as pd

    from ngio.images import TableWithExtras, concatenate_tables, conctatenate_tables
    from ngio.tables import FeatureTable

    def make():
        data = pd.DataFrame({"x": [1, 2], "label": [1, 2]})
        return [TableWithExtras(table=FeatureTable(table_data=data), extras={"a": "b"})]

    expected = concatenate_tables(make()).dataframe
    with pytest.warns(NgioDeprecationWarning, match="concatenate_tables"):
        got = conctatenate_tables(make()).dataframe
    assert got.equals(expected)


class TestMaskedLabelPixelSize:
    """`get_masked_label` resolved the masking label at the wrong level."""

    @pytest.fixture
    def container(self, tmp_path) -> OmeZarrContainer:
        ome = create_empty_ome_zarr(
            store=tmp_path / "masked.zarr",
            shape=(64, 64),
            pixelsize=0.5,
            levels=3,
        )
        label = ome.derive_label(name="nuclei")
        arr = np.zeros((64, 64), dtype="uint32")
        arr[:32, :32] = 1
        arr[32:, 32:] = 2
        label.set_array(arr)
        label.consolidate()
        ome.build_masking_roi_table("nuclei")
        return ome

    @pytest.mark.parametrize("path", ["0", "1", "2"])
    def test_masking_label_matches_requested_level(self, container, path):
        masked = container.get_masked_label(
            label_name="nuclei", masking_label_name="nuclei", path=path
        )
        assert masked.pixel_size == masked._label.pixel_size

    def test_masked_image_and_label_agree(self, container):
        image = container.get_masked_image(masking_label_name="nuclei", path="2")
        label = container.get_masked_label(
            label_name="nuclei", masking_label_name="nuclei", path="2"
        )
        assert image._label.pixel_size == label._label.pixel_size


class TestMaxWorkers:
    @pytest.fixture
    def plate(self, tmp_path):
        from ngio import create_empty_plate, open_ome_zarr_plate

        store = tmp_path / "mw.zarr"
        plate = create_empty_plate(store=store, name="p")
        for row in "ABCD":
            for col in range(1, 4):
                plate.add_image(row=row, column=col, image_path="0")
                create_empty_ome_zarr(
                    store=plate.get_image_store(row=row, column=col, image_path="0"),
                    shape=(16, 16),
                    pixelsize=0.5,
                    overwrite=True,
                )
        return open_ome_zarr_plate(store)

    @pytest.mark.parametrize("max_workers", [None, 1, 4])
    def test_get_images_is_independent_of_max_workers(self, plate, max_workers):
        serial = plate.get_images()
        assert list(plate.get_images(max_workers=max_workers)) == list(serial)

    @pytest.mark.parametrize("max_workers", [None, 1, 4])
    def test_get_wells_is_independent_of_max_workers(self, plate, max_workers):
        serial = plate.get_wells()
        assert list(plate.get_wells(max_workers=max_workers)) == list(serial)

    def test_async_methods_are_deprecated(self, plate):
        with pytest.warns(NgioDeprecationWarning, match="get_images"):
            coro = plate.get_images_async()
        assert list(asyncio.run(coro)) == list(plate.get_images())

        with pytest.warns(NgioDeprecationWarning, match="get_wells"):
            coro = plate.get_wells_async()
        assert list(asyncio.run(coro)) == list(plate.get_wells())

        with pytest.warns(NgioDeprecationWarning, match="images_paths"):
            coro = plate.images_paths_async()
        assert asyncio.run(coro) == plate.images_paths()
