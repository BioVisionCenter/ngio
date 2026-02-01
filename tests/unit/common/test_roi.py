import pytest

from ngio import PixelSize
from ngio.common import Roi
from ngio.utils import NgioValueError


def test_basic_rois_ops():
    roi = Roi.from_values(
        name="test",
        slices={
            "x": (0, 1),
            "y": (0, 1),
            "z": (0, 1),
        },
        space="world",
        other="other",  # type: ignore
        label=1,
    )

    slixe_x = roi.get("x")
    assert slixe_x is not None
    assert slixe_x.axis_name == "x"
    assert slixe_x.start == 0
    assert slixe_x.length == 1

    pixel_size = PixelSize(x=1.0, y=1.0, z=1.0)
    raster_roi = roi.to_pixel(pixel_size)
    assert roi.__str__()
    assert roi.__repr__()

    assert raster_roi.to_slicing_dict(pixel_size=pixel_size) == {
        "x": slice(0.0, 1.0),
        "y": slice(0.0, 1.0),
        "z": slice(0.0, 1.0),
    }
    assert roi.model_extra is not None
    assert roi.model_extra["other"] == "other"

    world_roi_2 = raster_roi.to_world(pixel_size)

    x_slice_2 = world_roi_2.get("x")
    assert x_slice_2 is not None
    assert x_slice_2.axis_name == "x"
    assert x_slice_2.start == 0
    assert x_slice_2.length == 1

    y_slice_2 = world_roi_2.get("y")
    assert y_slice_2 is not None
    assert y_slice_2.axis_name == "y"
    assert y_slice_2.start == 0
    assert y_slice_2.length == 1
    assert world_roi_2.other == "other"  # type: ignore

    roi_zoomed = roi.zoom(2.0)
    with pytest.raises(ValueError):
        roi.zoom(-1.0)

    assert roi_zoomed.to_slicing_dict(pixel_size) == {
        "x": slice(0.0, 2.0),
        "y": slice(0.0, 2.0),
        "z": slice(0.0, 1.0),
    }

    roi2 = Roi.from_values(
        name="test2",
        slices={
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (0.0, 1.0),
        },
        space="world",
        # type: ignore
        label=1,
    )
    roi_i = roi.intersection(roi2)
    assert roi_i is not None
    assert roi_i.label == 1

    roi2.label = 2
    with pytest.raises(NgioValueError):
        roi.intersection(roi2)


@pytest.mark.parametrize(
    "roi_ref,roi_other,expected_intersection,expected_name",
    [
        (
            # Basic intersection
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "z": (0.0, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name="other",
                slices={
                    "x": (0.5, 1.0),
                    "y": (0.5, 1.0),
                    "z": (0.5, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name="ref:other",
                slices={
                    "x": (0.5, 0.5),
                    "y": (0.5, 0.5),
                    "z": (0.5, 0.5),
                },
                space="world",
            ),
            "ref:other",
        ),
        (
            # No intersection
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "z": (0.0, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name="other",
                slices={
                    "x": (2.0, 1.0),
                    "y": (2.0, 1.0),
                    "z": (2.0, 1.0),
                },
                space="world",
            ),
            None,
            "",
        ),
        (
            # Intersection with z=None (expected behaves like infinite z)
            # t=None (expected behaves like infinite t)
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name=None,
                slices={
                    "x": (0.5, 1.0),
                    "y": (0.5, 1.0),
                    "z": (-1.0, 2.0),
                    "t": (0.0, 2.0),
                },
                space="world",
                unit="micrometer",
            ),
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.5, 0.5),
                    "y": (0.5, 0.5),
                    "z": (-1.0, 2.0),
                    "t": (0.0, 2.0),
                },
                space="world",
            ),
            "ref",
        ),
    ],
)
def test_rois_intersection(
    roi_ref: Roi,
    roi_other: Roi,
    expected_intersection: Roi | None,
    expected_name: str,
):
    intersection = roi_ref.intersection(roi_other)
    if expected_intersection is None:
        assert intersection is None
    else:
        assert intersection is not None
        assert intersection.name == expected_name
        assert intersection.get("x") == expected_intersection.get("x")
        assert intersection.get("y") == expected_intersection.get("y")
        assert intersection.get("z") == expected_intersection.get("z")

        assert intersection.get("t") == expected_intersection.get("t")


@pytest.mark.parametrize(
    "roi_ref,roi_other,expected_union,expected_name",
    [
        (
            # Basic intersection
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "z": (0.0, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name="other",
                slices={
                    "x": (0.5, 1.0),
                    "y": (0.5, 1.0),
                    "z": (0.5, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name="ref:other",
                slices={
                    "x": (0.0, 1.5),
                    "y": (0.0, 1.5),
                    "z": (0.0, 1.5),
                },
                space="world",
            ),
            "ref:other",
        ),
        (
            # Intersection with z=None (expected behaves like infinite z)
            # t=None (expected behaves like infinite t)
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                },
                space="world",
            ),
            Roi.from_values(
                name=None,
                slices={
                    "x": (0.5, 1.0),
                    "y": (0.5, 1.0),
                    "z": (-1.0, 2.0),
                    "t": (0.0, 2.0),
                },
                space="world",
                unit="micrometer",
            ),
            Roi.from_values(
                name="ref",
                slices={
                    "x": (0.0, 1.5),
                    "y": (0.0, 1.5),
                    "z": (-1.0, 2.0),
                    "t": (0.0, 2.0),
                },
                space="world",
            ),
            "ref",
        ),
    ],
)
def test_rois_union(
    roi_ref: Roi,
    roi_other: Roi,
    expected_union: Roi,
    expected_name: str,
):
    union = roi_ref.union(roi_other)
    assert union is not None
    assert union.name == expected_name
    assert union.get("x") == expected_union.get("x")
    assert union.get("y") == expected_union.get("y")
    assert union.get("z") == expected_union.get("z")
    assert union.get("t") == expected_union.get("t")
