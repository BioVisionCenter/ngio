import pytest

from ngio import PixelSize
from ngio.common import Roi
from ngio.utils import NgioValueError


def test_basic_rois_ops():
    roi = Roi(
        name="test",
        x=0.0,
        y=0.0,
        z=0.0,
        x_length=1.0,
        y_length=1.0,
        z_length=1.0,
        label=1,
        unit="micrometer",  # type: ignore
        other="other",  # type: ignore
    )

    assert roi.x == 0.0

    pixel_size = PixelSize(x=1.0, y=1.0, z=1.0)
    raster_roi = roi.to_roi_pixels(pixel_size)
    assert roi.__str__()
    assert roi.__repr__()

    assert raster_roi.to_slicing_dict(pixel_size=pixel_size) == {
        "x": slice(0, 1),
        "y": slice(0, 1),
        "z": slice(0, 1),
        "t": slice(None),
    }
    assert roi.model_extra is not None
    assert roi.model_extra["other"] == "other"

    world_roi_2 = raster_roi.to_roi(pixel_size)

    assert world_roi_2.x == 0.0
    assert world_roi_2.y == 0.0
    assert world_roi_2.z == 0.0
    assert world_roi_2.x_length == 1.0
    assert world_roi_2.y_length == 1.0
    assert world_roi_2.z_length == 1.0
    assert world_roi_2.other == "other"  # type: ignore

    roi_zoomed = roi.zoom(2.0)
    with pytest.raises(ValueError):
        roi.zoom(-1.0)

    assert roi_zoomed.to_slicing_dict(pixel_size) == {
        "x": slice(0, 2),
        "y": slice(0, 2),
        "z": slice(0, 1),
        "t": slice(None),
    }

    roi2 = Roi(
        name="test2",
        x=0.0,
        y=0.0,
        z=0.0,
        x_length=1.0,
        y_length=1.0,
        z_length=1.0,
        unit="micrometer",  # type: ignore
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
            Roi(
                name="ref",
                x=0.0,
                y=0.0,
                z=0.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name="other",
                x=0.5,
                y=0.5,
                z=0.5,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name="ref:other",
                x=0.5,
                y=0.5,
                z=0.5,
                x_length=0.5,
                y_length=0.5,
                z_length=0.5,
                unit="micrometer",  # type: ignore
            ),
            "ref:other",
        ),
        (
            # No intersection
            Roi(
                name="ref",
                x=0.0,
                y=0.0,
                z=0.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name="other",
                x=2.0,
                y=2.0,
                z=2.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            ),
            None,
            "",
        ),
        (
            # Intersection with z=None (expected behaves like infinite z)
            # t=None (expected behaves like infinite t)
            Roi(
                name="ref",
                x=0.0,
                y=0.0,
                z=None,
                t=0,
                x_length=2.0,
                y_length=2.0,
                z_length=None,
                t_length=2.0,
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name=None,
                x=-1.0,
                y=-1.0,
                z=-1.0,
                t=None,
                x_length=2.0,
                y_length=2.0,
                z_length=2.0,
                t_length=None,
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name="ref",
                x=0.0,
                y=0.0,
                z=-1.0,
                t=0,
                x_length=1.0,
                y_length=1.0,
                z_length=2.0,
                t_length=2.0,
                unit="micrometer",  # type: ignore
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
        assert intersection.x == expected_intersection.x
        assert intersection.y == expected_intersection.y
        assert intersection.z == expected_intersection.z
        assert intersection.x_length == expected_intersection.x_length
        assert intersection.y_length == expected_intersection.y_length
        assert intersection.z_length == expected_intersection.z_length
