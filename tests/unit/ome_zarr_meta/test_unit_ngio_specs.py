import numpy as np
import pytest

from ngio.ome_zarr_meta.ngio_specs import (
    AxesHandler,
    AxesSetup,
    Axis,
    AxisType,
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    Dataset,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgioColors,
    NgioImageMeta,
    NgioLabelMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs._channels import valid_hex_color


@pytest.mark.parametrize(
    "axes, axes_setup, allow_non_canonical_axes, strict_canonical_order",
    [
        (
            ["t", "c", "z", "y", "x"],
            None,
            False,
            True,
        ),
        (
            ["c", "t", "z", "y", "x"],
            None,
            False,
            False,
        ),
        (
            ["c", "t", "z", "y", "X"],
            AxesSetup(x="X"),
            False,
            False,
        ),
        (
            ["y", "X"],
            AxesSetup(x="X"),
            False,
            False,
        ),
        (
            ["weird", "y", "X"],
            AxesSetup(x="X", others=["weird"]),
            True,
            False,
        ),
    ],
)
def test_axes_base(axes, axes_setup, allow_non_canonical_axes, strict_canonical_order):
    _axes = [Axis(name=name) for name in axes]
    mapper = AxesHandler(
        axes=_axes,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )
    for i, ax in enumerate(axes):
        assert mapper.get_index(ax) == i

    assert len(mapper.axes) == len(axes)


@pytest.mark.parametrize(
    "canonical_name, axis_type, unit, expected_type, expected_unit",
    [
        ("x", AxisType.space, None, AxisType.space, "micrometer"),
        ("x", AxisType.time, "second", AxisType.space, "second"),
        ("t", AxisType.time, None, AxisType.time, "second"),
        ("c", AxisType.channel, None, AxisType.channel, None),
    ],
)
def test_axis_cast(canonical_name, axis_type, unit, expected_type, expected_unit):
    ax = Axis(
        name="temp",
        unit=unit,
        axis_type=axis_type,
    )
    ax = ax.canonical_axis_cast(canonical_name)
    assert ax.axis_type == expected_type
    assert ax.unit == expected_unit


def test_axes_fail():
    with pytest.raises(ValueError):
        AxesHandler(
            axes=[Axis(name="x")],
            axes_setup=AxesSetup(x="X"),
            allow_non_canonical_axes=False,
            strict_canonical_order=False,
        )

    with pytest.raises(ValueError):
        AxesHandler(
            axes=[Axis(name="x")],
            axes_setup=AxesSetup(x="x"),
            allow_non_canonical_axes=True,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesHandler(
            axes=[
                Axis(name="x"),
                Axis(name="x"),
            ],
            axes_setup=None,
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesHandler(
            axes=[
                Axis(name="x"),
                Axis(name="z"),
            ],
            axes_setup=None,
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesHandler(
            axes=[
                Axis(name="weird"),
                Axis(name="y"),
                Axis(name="x"),
            ],
            axes_setup=AxesSetup(others=["weird"]),
            allow_non_canonical_axes=False,
            strict_canonical_order=False,
        )


def test_pixel_size():
    ps_dict = {"x": 0.5, "y": 0.5, "z": 1.0, "t": 1.0}
    ps_1 = PixelSize(**ps_dict, space_unit=DefaultSpaceUnit, time_unit=DefaultTimeUnit)
    assert ps_1.as_dict() == ps_dict
    assert ps_1.zyx == (1.0, 0.5, 0.5)
    assert ps_1.yx == (0.5, 0.5)
    assert ps_1.voxel_volume == 0.25
    assert ps_1.xy_plane_area == 0.25
    assert ps_1.time_spacing == 1.0

    ps_2 = PixelSize(x=0.5, y=0.5, z=1.0, t=1.0)
    np.testing.assert_allclose(ps_1.distance(ps_2), 0.0)
    ps_3 = PixelSize(x=1.0, y=1.0, z=1.0, t=1.0)
    np.testing.assert_allclose(ps_1.distance(ps_3), np.sqrt(2.0) / 2)

    # Test comparison
    p1 = PixelSize(x=1, y=1, z=0.1243532)
    p2 = PixelSize(x=1, y=1, z=0.1243532)
    assert p1 == p2

    p_small = PixelSize(x=0.1, y=0.1, z=0.1)
    p_large = PixelSize(x=2, y=2, z=2)
    assert p_small < p_large


def test_dataset():
    axes = [
        Axis(name="t", axis_type=AxisType.time, unit=DefaultTimeUnit),
        Axis(name="c", axis_type=AxisType.channel),
        Axis(name="z"),
        Axis(name="y"),
        Axis(name="x"),
    ]

    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )

    scale = [1.0, 1.0, 1.0, 0.5, 0.5]
    translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    ds = Dataset(
        path="0",
        axes_handler=axes_handler,
        scale=scale,
        translation=translation,
    )

    assert ds.path == "0"
    assert ds.axes_handler.get_index("x") == 4
    assert ds.scale == tuple(scale)
    assert ds.translation == tuple(translation)

    ps = ds.pixel_size
    assert ps.x == 0.5
    assert ps.y == 0.5
    assert ps.z == 1.0
    assert ps.t == 1.0


def test_dataset_fail():
    axes = [
        Axis(name="y", unit="centimeter"),
        Axis(name="x", unit="micrometer"),
    ]
    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )
    ds = Dataset(
        path="0",
        axes_handler=axes_handler,
        scale=[0.5, 0.5],
        translation=[0.0, 0.0],
    )

    assert ds.axes_handler.time_unit is None

    with pytest.raises(ValueError):
        assert ds.axes_handler.space_unit == "micrometer"


def test_channels():
    channels = ChannelsMeta.default_init(
        labels=["DAPI", "GFP", "RFP"],
    )
    assert len(channels.channels) == 3
    assert channels.channels[0].label == "DAPI"
    assert channels.channels[0].wavelength_id == "DAPI"
    assert channels.channels[0].channel_visualisation.color == NgioColors.dapi.value

    channels = ChannelsMeta.default_init(labels=4)
    assert len(channels.channels) == 4
    assert channels.channels[0].label == "channel_0"
    assert channels.channels[0].wavelength_id == "channel_0"
    assert channels.channels[0].channel_visualisation.color == "00FFFF"

    channels = ChannelsMeta.default_init(
        labels=["DAPI", "GFP", "RFP"],
        wavelength_id=["A01_C01", "A02_C02", "A03_C03"],
        colors=["00FF00", "FF0000", "00FFFF"],
        active=[True, False, True],
        end=[100, 200, 300],
        start=[0, 100, 200],
        data_type="float",
    )
    assert len(channels.channels) == 3
    assert channels.channels[0].label == "DAPI"
    assert channels.channels[0].wavelength_id == "A01_C01"
    assert channels.channels[0].channel_visualisation.color == "00FF00"

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[])

    class Mock:
        pass

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[Mock()])  # type: ignore

    channel = Channel.default_init(label="DAPI", wavelength_id="A01_C01")
    ChannelsMeta(channels=[channel])

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=["DAPI", "DAPI"])

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[channel, channel])  # type: ignore

    with pytest.raises(ValueError):
        Channel.default_init(label="DAPI", data_type="color")


def test_ngio_colors():
    assert NgioColors.semi_random_pick(channel_name="DAPI") == NgioColors.dapi
    assert NgioColors.semi_random_pick(channel_name="channel_dapi") == NgioColors.dapi
    assert valid_hex_color(NgioColors.semi_random_pick(channel_name=None))

    for channel, expected in zip(
        ["channel_0", "channel_1", "channel_2", "channel_3"],
        [NgioColors.cyan, NgioColors.magenta, NgioColors.yellow, NgioColors.green],
        strict=True,
    ):
        assert NgioColors.semi_random_pick(channel_name=channel) == expected

    for non_hex_color in ["00000000", "not a color"]:
        assert not valid_hex_color(non_hex_color)

    for color in [None, NgioColors.cyan]:
        ChannelVisualisation.default_init(color=color)

    ChannelVisualisation(color=NgioColors.cyan)

    with pytest.raises(ValueError):
        ChannelVisualisation.default_init(color=[])  # type: ignore


def test_image_meta():
    axes = [
        Axis(name="t", axis_type=AxisType.time, unit=DefaultSpaceUnit),
        Axis(name="c", axis_type=AxisType.channel),
        Axis(name="z"),
        Axis(name="y"),
        Axis(name="x"),
    ]

    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )
    translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            axes_handler=axes_handler,
            scale=scale,
            translation=translation,
        )
        datasets.append(ds)
        scale = [s * f for s, f in zip(scale, [1, 1, 1, 2, 2], strict=True)]

    image_meta = NgioImageMeta(version="0.4", name="test", datasets=datasets)

    image_meta.init_channels(labels=["DAPI", "GFP", "RFP"])

    assert image_meta.levels == 4
    assert image_meta.name == "test"
    assert image_meta.version == "0.4"
    assert len(image_meta.scaling_factor()) == 5
    np.testing.assert_allclose(image_meta.scaling_factor(), [1, 1, 1, 2, 2])
    assert image_meta.get_dataset(path="0").path == "0"
    assert image_meta.get_dataset(path="1").path == "1"
    assert image_meta.get_dataset().path == "0"
    assert image_meta.get_dataset(pixel_size=datasets[-1].pixel_size).path == "3"
    channels_meta = image_meta.channels_meta
    assert channels_meta is not None
    assert channels_meta.get_channel_idx(channel_label="DAPI") == 0
    assert channels_meta.get_channel_idx(wavelength_id="DAPI") == 0
    assert channels_meta.channel_labels == ["DAPI", "GFP", "RFP"]


def test_label_meta():
    axes = [
        Axis(name="t", axis_type=AxisType.time, unit=DefaultSpaceUnit),
        Axis(name="z"),
        Axis(name="y"),
        Axis(name="x"),
    ]
    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )
    translation = [0.0, 0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            axes_handler=axes_handler,
            scale=scale,
            translation=translation,
        )
        datasets.append(ds)
        scale = [s * f for s, f in zip(scale, [1, 1, 2, 2], strict=True)]

    label_meta = NgioLabelMeta(
        version="0.4",
        name="test",
        datasets=datasets,
    )
    assert label_meta.source_image == "../../"
    assert label_meta.levels == 4
    assert label_meta.name == "test"
    assert label_meta.version == "0.4"
    np.testing.assert_allclose(label_meta.scaling_factor(), [1, 1, 2, 2])
    assert label_meta.get_dataset(path="0").path == "0"
    assert label_meta.get_dataset(path="1").path == "1"
    assert label_meta.get_dataset().path == "0"
    assert label_meta.get_dataset(pixel_size=datasets[-1].pixel_size).path == "3"


def test_channels_label_meta():
    axes = [
        Axis(name="t", axis_type=AxisType.time, unit=DefaultSpaceUnit),
        Axis(name="c"),
        Axis(name="z"),
        Axis(name="y"),
        Axis(name="x"),
    ]
    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )
    translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            axes_handler=axes_handler,
            scale=scale,
            translation=translation,
        )
        datasets.append(ds)
        scale = [s * f for s, f in zip(scale, [1, 1, 1, 2, 2], strict=True)]

    _ = NgioLabelMeta(version="0.4", name="test", datasets=datasets)
