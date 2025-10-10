import pytest

from ngio import PixelSize
from ngio.common import Roi, RoiSlice
from ngio.utils import NgioValueError


def test_basic_rois_ops():
    roi = Roi(
        name="test",
        x=RoiSlice(start=0.0, length=1.0),
        y=RoiSlice(start=0.0, length=1.0),
        z=RoiSlice(start=0.0, length=1.0),
        label=1,
        unit="micrometer",
        other="other",  # type: ignore
    )

    assert roi.x.start == 0.0

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
    assert world_roi_2.z is not None

    assert world_roi_2.x.start == 0.0
    assert world_roi_2.y.start == 0.0
    assert world_roi_2.z.start == 0.0

    assert world_roi_2.x.length == 1.0
    assert world_roi_2.y.length == 1.0
    assert world_roi_2.z.length == 1.0
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
        x=RoiSlice(start=0, length=1.0),
        y=RoiSlice(start=0, length=1.0),
        z=RoiSlice(start=0, length=1.0),
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
                x=RoiSlice(start=0.0, length=1.0),
                y=RoiSlice(start=0.0, length=1.0),
                z=RoiSlice(start=0.0, length=1.0),
                unit="micrometer",
            ),
            Roi(
                name="other",
                x=RoiSlice(start=0.5, length=1.0),
                y=RoiSlice(start=0.5, length=1.0),
                z=RoiSlice(start=0.5, length=1.0),
                unit="micrometer",
            ),
            Roi(
                name="ref:other",
                x=RoiSlice(start=0.5, length=0.5),
                y=RoiSlice(start=0.5, length=0.5),
                z=RoiSlice(start=0.5, length=0.5),
                unit="micrometer",
            ),
            "ref:other",
        ),
        (
            # No intersection
            Roi(
                name="ref",
                x=RoiSlice(start=0.0, length=1.0),
                y=RoiSlice(start=0.0, length=1.0),
                z=RoiSlice(start=0.0, length=1.0),
                unit="micrometer",  # type: ignore
            ),
            Roi(
                name="other",
                x=RoiSlice(start=2.0, length=1.0),
                y=RoiSlice(start=2.0, length=1.0),
                z=RoiSlice(start=2.0, length=1.0),
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
                x=RoiSlice(start=0.0, length=2.0),
                y=RoiSlice(start=0.0, length=2.0),
                z=None,
                t=RoiSlice(start=0, length=2),
                unit="micrometer",
            ),
            Roi(
                name=None,
                x=RoiSlice(start=-1.0, length=2.0),
                y=RoiSlice(start=-1.0, length=2.0),
                z=RoiSlice(start=-1.0, length=2.0),
                t=None,
                unit="micrometer",
            ),
            Roi(
                name="ref",
                x=RoiSlice(start=0.0, length=1.0),
                y=RoiSlice(start=0.0, length=1.0),
                z=RoiSlice(start=-1.0, length=2.0),
                t=RoiSlice(start=0, length=2),
                unit="micrometer",
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
        assert intersection.x.start == expected_intersection.x.start
        assert intersection.y.start == expected_intersection.y.start
        assert intersection.z is not None
        assert expected_intersection.z is not None
        assert intersection.z.start == expected_intersection.z.start
        assert intersection.y == expected_intersection.y
        assert intersection.z == expected_intersection.z
        assert intersection.x.length == expected_intersection.x.length
        assert intersection.y.length == expected_intersection.y.length
        assert intersection.z.length == expected_intersection.z.length
