import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.ome_zarr_meta import (
    NgioImageMeta,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    NgioPlateMeta,
    NgioWellMeta,
)
from ngio.ome_zarr_meta.ngio_specs import AxesSetup
from ngio.ome_zarr_meta.v05._v05_spec import (
    ngio_to_v05_image_meta,
    ngio_to_v05_label_meta,
    ngio_to_v05_labels_group_meta,
    ngio_to_v05_plate_meta,
    ngio_to_v05_well_meta,
    v05_to_ngio_image_meta,
    v05_to_ngio_label_meta,
    v05_to_ngio_labels_group_meta,
    v05_to_ngio_plate_meta,
    v05_to_ngio_well_meta,
)
from ngio.utils import ZarrGroupHandler


def _base_multiscale(name="image") -> dict:
    return {
        "axes": [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
            },
            {
                "path": "1",
                "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}],
            },
        ],
        "name": name,
    }


def _image_attrs(multiscales=None, omero=None) -> dict:
    ome: dict = {
        "version": "0.5",
        "multiscales": multiscales if multiscales is not None else [_base_multiscale()],
    }
    if omero is not None:
        ome["omero"] = omero
    return {"ome": ome}


def test_image_round_trip():
    omero = {
        "channels": [
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 100, "min": 0, "max": 255},
                "label": "DAPI",
                "wavelength_id": "A01_C01",
                "active": False,
            },
            # bare channel: label/wavelength_id/active fall back to defaults
            {
                "color": "FF0000",
                "window": {"start": 0, "end": 100, "min": 0, "max": 255},
            },
        ]
    }
    input_attrs = _image_attrs(omero=omero)

    ngio_image = v05_to_ngio_image_meta(input_attrs, axes_setup=AxesSetup())
    assert isinstance(ngio_image, NgioImageMeta)
    assert ngio_image.version == "0.5"
    assert ngio_image.name == "image"

    channels = ngio_image.channels_meta
    assert channels is not None
    assert channels.channels[0].label == "DAPI"
    assert channels.channels[0].wavelength_id == "A01_C01"
    assert channels.channels[0].channel_visualisation.active is False
    # defaults for the bare channel
    assert channels.channels[1].label == channels.channels[1].wavelength_id
    assert channels.channels[1].channel_visualisation.active is True

    output_attrs = ngio_to_v05_image_meta(ngio_image)
    ngio_image_2 = v05_to_ngio_image_meta(output_attrs, axes_setup=AxesSetup())

    assert ngio_image_2.name == ngio_image.name
    for ds_1, ds_2 in zip(ngio_image.datasets, ngio_image_2.datasets, strict=True):
        assert ds_1.path == ds_2.path
        assert ds_1.scale == ds_2.scale
        assert ds_1.translation == ds_2.translation
    channels_2 = ngio_image_2.channels_meta
    assert channels_2 is not None
    assert [c.label for c in channels_2.channels] == [
        c.label for c in channels.channels
    ]


def test_image_multiple_multiscales_not_supported():
    attrs = _image_attrs(multiscales=[_base_multiscale(), _base_multiscale()])
    with pytest.raises(NotImplementedError):
        v05_to_ngio_image_meta(attrs, axes_setup=AxesSetup())


def test_image_non_string_multiscale_name_is_stringified():
    multiscale = _base_multiscale(name=123)
    ngio_image = v05_to_ngio_image_meta(
        _image_attrs(multiscales=[multiscale]), axes_setup=AxesSetup()
    )
    assert ngio_image.name == "123"


def test_image_non_string_axis_unit_is_stringified():
    multiscale = _base_multiscale()
    multiscale["axes"][0]["unit"] = 5
    ngio_image = v05_to_ngio_image_meta(
        _image_attrs(multiscales=[multiscale]), axes_setup=AxesSetup()
    )
    y_axis = ngio_image.datasets[0].axes_handler.axes[0]
    assert y_axis.unit == "5"


def test_image_global_coordinate_transformations():
    multiscale = _base_multiscale()
    multiscale["coordinateTransformations"] = [{"type": "scale", "scale": [2.0, 2.0]}]
    ngio_image = v05_to_ngio_image_meta(
        _image_attrs(multiscales=[multiscale]), axes_setup=AxesSetup()
    )
    # the multiscale-level scale multiplies every dataset scale
    assert ngio_image.datasets[0].scale == (2.0, 2.0)
    assert ngio_image.datasets[1].scale == (4.0, 4.0)


def _label_attrs() -> dict:
    """Attrs of a real derived label, read back from an in-memory container."""
    store = MemoryStore()
    container = create_ome_zarr_from_array(
        store=store,
        array=np.zeros((16, 16), dtype="uint8"),
        pixelsize=1.0,
        axes_names=["y", "x"],
        levels=2,
        ngff_version="0.5",
    )
    container.derive_label("lbl")
    handler = ZarrGroupHandler(store=store, mode="r")
    return handler.group["labels/lbl"].attrs.asdict()


def test_label_round_trip():
    input_attrs = _label_attrs()

    ngio_label = v05_to_ngio_label_meta(input_attrs, axes_setup=AxesSetup())
    assert isinstance(ngio_label, NgioLabelMeta)
    assert ngio_label.version == "0.5"

    output_attrs = ngio_to_v05_label_meta(ngio_label)
    ngio_label_2 = v05_to_ngio_label_meta(output_attrs, axes_setup=AxesSetup())

    for ds_1, ds_2 in zip(ngio_label.datasets, ngio_label_2.datasets, strict=True):
        assert ds_1.path == ds_2.path
        assert ds_1.scale == ds_2.scale
    assert ngio_label.image_label == ngio_label_2.image_label


def test_label_multiple_multiscales_not_supported():
    attrs = _label_attrs()
    attrs["ome"]["multiscales"] = attrs["ome"]["multiscales"] * 2
    with pytest.raises(NotImplementedError):
        v05_to_ngio_label_meta(attrs, axes_setup=AxesSetup())


def test_well_round_trip():
    ngio_well = (
        NgioWellMeta.default_init(ngff_version="0.5")
        .add_image(path="0")
        .add_image(path="1", acquisition=0, strict=False)
    )
    output_attrs = ngio_to_v05_well_meta(ngio_well)
    ngio_well_2 = v05_to_ngio_well_meta(output_attrs)

    assert isinstance(ngio_well_2, NgioWellMeta)
    assert ngio_well_2.version == "0.5"
    assert ngio_well_2.paths() == ["0", "1"]
    assert ngio_well_2.get_image_acquisition_id("1") == 0


def test_plate_round_trip():
    ngio_plate = (
        NgioPlateMeta.default_init(name="plate", ngff_version="0.5")
        .add_well(row="A", column=1)
        .add_acquisition(acquisition_id=0, acquisition_name="acq")
    )
    output_attrs = ngio_to_v05_plate_meta(ngio_plate)
    ngio_plate_2 = v05_to_ngio_plate_meta(output_attrs)

    assert isinstance(ngio_plate_2, NgioPlateMeta)
    assert ngio_plate_2.version == "0.5"
    assert ngio_plate_2.wells_paths == ["A/01"]
    assert ngio_plate_2.acquisition_ids == [0]
    assert ngio_plate_2.plate.name == "plate"


def test_labels_group_round_trip():
    ngio_labels = NgioLabelsGroupMeta(labels=["a", "b"], version="0.5")
    output_attrs = ngio_to_v05_labels_group_meta(ngio_labels)
    ngio_labels_2 = v05_to_ngio_labels_group_meta(output_attrs)

    assert isinstance(ngio_labels_2, NgioLabelsGroupMeta)
    assert list(ngio_labels_2.labels) == ["a", "b"]
