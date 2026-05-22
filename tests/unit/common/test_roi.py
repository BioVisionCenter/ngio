import pytest

from ngio import PixelSize
from ngio.common import Roi, RoiSlice
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


# ---------------------------------------------------------------------------
# RoiSlice unit tests
# ---------------------------------------------------------------------------


def test_roi_slice_from_value_slice():
    s = RoiSlice.from_value("x", slice(2.0, 5.0))
    assert s.axis_name == "x"
    assert s.start == 2.0
    assert s.length == 3.0


def test_roi_slice_from_value_tuple():
    s = RoiSlice.from_value("y", (1.0, 4.0))
    assert s.start == 1.0
    assert s.length == 4.0


def test_roi_slice_from_value_float():
    s = RoiSlice.from_value("z", 3.0)
    assert s.start == 3.0
    assert s.length == 1


def test_roi_slice_from_value_int():
    s = RoiSlice.from_value("z", 7)
    assert s.start == 7
    assert s.length == 1


def test_roi_slice_from_value_roi_slice():
    original = RoiSlice(axis_name="x", start=1.0, length=2.0)
    result = RoiSlice.from_value("x", original)
    assert result is original


def test_roi_slice_from_value_invalid_type():
    with pytest.raises(TypeError):
        RoiSlice.from_value("x", "invalid")  # type: ignore


def test_roi_slice_end():
    s = RoiSlice(axis_name="x", start=2.0, length=3.0)
    assert s.end == 5.0


def test_roi_slice_end_none_when_unset():
    assert RoiSlice(axis_name="x").end is None
    assert RoiSlice(axis_name="x", start=1.0).end is None
    assert RoiSlice(axis_name="x", length=1.0).end is None


def test_roi_slice_to_slice():
    s = RoiSlice(axis_name="x", start=1.0, length=3.0)
    assert s.to_slice() == slice(1.0, 4.0)


def test_roi_slice_union():
    a = RoiSlice(axis_name="x", start=0.0, length=2.0)
    b = RoiSlice(axis_name="x", start=1.0, length=2.0)
    u = a.union(b)
    assert u.start == 0.0
    assert u.end == 3.0


def test_roi_slice_union_incompatible_axes():
    a = RoiSlice(axis_name="x", start=0.0, length=2.0)
    b = RoiSlice(axis_name="y", start=0.0, length=2.0)
    with pytest.raises(NgioValueError):
        a.union(b)


def test_roi_slice_intersection():
    a = RoiSlice(axis_name="x", start=0.0, length=3.0)
    b = RoiSlice(axis_name="x", start=2.0, length=3.0)
    i = a.intersection(b)
    assert i is not None
    assert i.start == 2.0
    assert i.length == 1.0


def test_roi_slice_intersection_none():
    a = RoiSlice(axis_name="x", start=0.0, length=1.0)
    b = RoiSlice(axis_name="x", start=2.0, length=1.0)
    assert a.intersection(b) is None


def test_roi_slice_intersection_incompatible_axes():
    a = RoiSlice(axis_name="x", start=0.0, length=1.0)
    b = RoiSlice(axis_name="y", start=0.0, length=1.0)
    with pytest.raises(NgioValueError):
        a.intersection(b)


def test_roi_slice_to_world():
    s = RoiSlice(axis_name="x", start=4.0, length=2.0)
    w = s.to_world(pixel_size=0.5)
    assert w.start == 2.0
    assert w.length == 1.0


def test_roi_slice_to_pixel():
    s = RoiSlice(axis_name="x", start=2.0, length=1.0)
    p = s.to_pixel(pixel_size=0.5)
    assert p.start == 4.0
    assert p.length == 2.0


def test_roi_slice_zoom_expand():
    s = RoiSlice(axis_name="x", start=1.0, length=2.0)
    z = s.zoom(2.0)
    assert z.length == 4.0


def test_roi_slice_zoom_shrink():
    s = RoiSlice(axis_name="x", start=0.0, length=4.0)
    z = s.zoom(0.5)
    assert z.length == 2.0


def test_roi_slice_zoom_no_length():
    s = RoiSlice(axis_name="x", start=0.0, length=None)
    assert s.zoom(2.0) is s


def test_roi_slice_zoom_invalid():
    s = RoiSlice(axis_name="x", start=0.0, length=2.0)
    with pytest.raises(NgioValueError):
        s.zoom(0.0)
    with pytest.raises(NgioValueError):
        s.zoom(-1.0)


# ---------------------------------------------------------------------------
# Roi.get_name
# ---------------------------------------------------------------------------


def test_roi_get_name_uses_name():
    roi = Roi.from_values(name="my_roi", slices={"x": (0, 1), "y": (0, 1)})
    assert roi.get_name() == "my_roi"


def test_roi_get_name_falls_back_to_label():
    roi = Roi.from_values(name=None, slices={"x": (0, 1), "y": (0, 1)}, label=42)
    assert roi.get_name() == "42"


def test_roi_get_name_falls_back_to_repr():
    roi = Roi.from_values(name=None, slices={"x": (0, 1), "y": (0, 1)})
    assert len(roi.get_name()) > 0


# ---------------------------------------------------------------------------
# Roi.to_world / to_pixel idempotency and missing pixel_size errors
# ---------------------------------------------------------------------------


def test_roi_to_world_idempotent():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)}, space="world")
    copy = roi.to_world()
    assert copy.space == "world"
    assert copy.get("x") == roi.get("x")


def test_roi_to_pixel_idempotent():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)}, space="pixel")
    copy = roi.to_pixel()
    assert copy.space == "pixel"
    assert copy.get("x") == roi.get("x")


def test_roi_to_pixel_requires_pixel_size():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)}, space="world")
    with pytest.raises(NgioValueError):
        roi.to_pixel()


def test_roi_to_world_requires_pixel_size():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)}, space="pixel")
    with pytest.raises(NgioValueError):
        roi.to_world()


# ---------------------------------------------------------------------------
# Roi.intersection / union space mismatch errors
# ---------------------------------------------------------------------------


def test_roi_intersection_space_mismatch():
    world = Roi.from_values(name="w", slices={"x": (0, 1), "y": (0, 1)}, space="world")
    pixel = Roi.from_values(name="p", slices={"x": (0, 1), "y": (0, 1)}, space="pixel")
    with pytest.raises(NgioValueError):
        world.intersection(pixel)


def test_roi_union_space_mismatch():
    world = Roi.from_values(name="w", slices={"x": (0, 1), "y": (0, 1)}, space="world")
    pixel = Roi.from_values(name="p", slices={"x": (0, 1), "y": (0, 1)}, space="pixel")
    with pytest.raises(NgioValueError):
        world.union(pixel)


# ---------------------------------------------------------------------------
# Roi.get with default and Roi.__getitem__
# ---------------------------------------------------------------------------


def test_roi_get_returns_none_when_missing():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    assert roi.get("z") is None


def test_roi_get_returns_custom_default_when_missing():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    fallback = RoiSlice(axis_name="z", start=0.0, length=10.0)
    result = roi.get("z", default=fallback)
    assert result is fallback


def test_roi_get_returns_slice_when_present():
    roi = Roi.from_values(name="r", slices={"x": (3.0, 5.0), "y": (0, 1)})
    result = roi.get("x")
    assert result is not None
    assert result.start == 3.0
    assert result.length == 5.0


def test_roi_getitem_returns_slice():
    roi = Roi.from_values(name="r", slices={"x": (3.0, 5.0), "y": (0, 1)})
    assert roi["x"].start == 3.0
    assert roi["x"].length == 5.0


def test_roi_getitem_raises_on_missing_axis():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    with pytest.raises(KeyError):
        _ = roi["z"]


# ---------------------------------------------------------------------------
# Roi.update_slice
# ---------------------------------------------------------------------------


def test_roi_update_slice_replaces_existing():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    updated = roi.update_slice("x", (5.0, 3.0))
    x = updated.get("x")
    assert x is not None
    assert x.start == 5.0
    assert x.length == 3.0
    # other axes unchanged
    assert updated.get("y") == roi.get("y")


def test_roi_update_slice_adds_new_axis():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    updated = roi.update_slice("z", (2.0, 4.0))
    z = updated.get("z")
    assert z is not None
    assert z.start == 2.0
    assert z.length == 4.0
    # original axes still present
    assert updated.get("x") == roi.get("x")
    assert updated.get("y") == roi.get("y")


def test_roi_update_slice_with_roi_slice():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    new_slice = RoiSlice(axis_name="x", start=10.0, length=5.0)
    updated = roi.update_slice("x", new_slice)
    assert updated.get("x") == new_slice


# ---------------------------------------------------------------------------
# Roi.remove_slice
# ---------------------------------------------------------------------------


def test_roi_remove_slice():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1), "z": (0, 1)})
    trimmed = roi.remove_slice("z")
    assert trimmed.get("z") is None
    assert trimmed.get("x") == roi.get("x")
    assert trimmed.get("y") == roi.get("y")


def test_roi_remove_slice_missing_axis():
    roi = Roi.from_values(name="r", slices={"x": (0, 1), "y": (0, 1)})
    with pytest.raises(NgioValueError):
        roi.remove_slice("z")


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
