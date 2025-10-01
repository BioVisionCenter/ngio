from ngio import PixelSize
from ngio.common import Roi


def test_rois():
    roi = Roi(
        name="test",
        x=0.0,
        y=0.0,
        z=0.0,
        x_length=1.0,
        y_length=1.0,
        z_length=1.0,
        unit="micrometer",  # type: ignore
        other="other",  # type: ignore
    )

    assert roi.x == 0.0

    pixel_size = PixelSize(x=1.0, y=1.0, z=1.0)
    raster_roi = roi.to_roi_pixels(pixel_size)

    assert raster_roi.to_slicing_dict() == {
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
