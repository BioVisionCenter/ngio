"""Object detection over tiles: boxes to world coordinates, NMS, one table."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import NmsConfig, ObjectDetectionIterator, create_ome_zarr_from_array
from ngio.iterators import ThreadedMapper
from ngio.utils import NgioValueError


def _image(data: np.ndarray, axes_names: str = "yx", pixelsize: float = 1.0):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=data,
        pixelsize=pixelsize,
        axes_names=axes_names,
        levels=1,
    )
    return ome_zarr, ome_zarr.get_image()


def _bright_box_detector(patch: np.ndarray, roi):
    """One box around everything bright in the patch, confidence = coverage.

    Content-driven, so every tile reports what it actually saw — which is how
    padded tiles produce genuine duplicates for NMS to resolve.
    """
    ys, xs = np.nonzero(patch > 128)
    if not len(ys):
        return {"x_min": [], "x_max": [], "y_min": [], "y_max": [], "confidence": []}
    return {
        "x_min": [int(xs.min())],
        "x_max": [int(xs.max()) + 1],
        "y_min": [int(ys.min())],
        "y_max": [int(ys.max()) + 1],
        "confidence": [float((patch > 128).mean())],
    }


def _two_blobs():
    """One blob crossing the x=32 tile seam, one wholly inside a tile."""
    data = np.zeros((64, 64), dtype="uint8")
    data[10:20, 26:38] = 255
    data[40:50, 8:18] = 255
    return data


def test_seam_duplicates_are_suppressed_into_one_object():
    _, image = _image(_two_blobs())
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    table = iterator.detect(_bright_box_detector)

    frame = table.dataframe
    # RoiTable indexes by name; the numeric object id rides in `label`.
    assert sorted(frame.index.tolist()) == ["1", "2"], "ids are compacted to 1..N"
    assert sorted(frame["label"].tolist()) == [1, 2]
    rois = {roi.label: roi for roi in table.rois()}
    seam = rois[1]
    assert (seam["x"].start, seam["x"].length) == (26.0, 12.0)
    assert (seam["y"].start, seam["y"].length) == (10.0, 10.0)
    other = rois[2]
    assert (other["x"].start, other["x"].length) == (8.0, 10.0)
    assert (other["y"].start, other["y"].length) == (40.0, 10.0)


def test_detect_parallel_matches_serial():
    _, image = _image(_two_blobs())
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    serial = iterator.detect(_bright_box_detector)
    threaded = iterator.detect(_bright_box_detector, mapper=ThreadedMapper(4))
    assert threaded.dataframe.equals(serial.dataframe)


def test_boxes_land_in_world_coordinates():
    """Patch-local pixels + padded tile origin + pixel size = world coords."""
    data = np.zeros((64, 64), dtype="uint8")
    data[40:52, 36:44] = 255  # wholly inside tile (1, 1)
    _, image = _image(data, pixelsize=0.5)
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=4, y=4)
    )

    table = iterator.detect(_bright_box_detector)

    boxes = [
        roi
        for roi in table.rois()
        if (roi["x"].start, roi["y"].start) == (36 * 0.5, 40 * 0.5)
    ]
    assert boxes, "the box must sit at the blob's world position"
    roi = boxes[0]
    assert (roi["x"].length, roi["y"].length) == (8 * 0.5, 12 * 0.5)
    # And back: pixel space returns the original box.
    pixel_roi = roi.to_pixel(pixel_size=image.pixel_size)
    assert (pixel_roi["x"].start, pixel_roi["y"].start) == (36.0, 40.0)


def test_halo_is_clipped_at_the_borders():
    """A corner blob: the halo clips at the borders, the box lands true."""
    data = np.zeros((64, 64), dtype="uint8")
    data[0:6, 0:6] = 255
    _, image = _image(data)
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=16, y=16)
    )

    table = iterator.detect(_bright_box_detector)

    roi = table.rois()[0]
    assert (roi["x"].start, roi["x"].length) == (0.0, 6.0)
    assert (roi["y"].start, roi["y"].length) == (0.0, 6.0)


def _fixed_boxes_factory(boxes):
    """A detector ignoring the pixels; used to drive NMS deterministically."""

    def detector(patch: np.ndarray, roi):
        return boxes

    return detector


def test_iou_threshold_separates_duplicates_from_neighbours():
    """IoU ≈ 0.47: suppressed at threshold 0.4, kept apart at 0.6."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = {
        "x_min": [0, 2],
        "x_max": [10, 12],
        "y_min": [0, 2],
        "y_max": [10, 12],
        "confidence": [0.9, 0.5],
    }

    merged = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.4)
    ).detect(_fixed_boxes_factory(boxes))
    assert len(merged.rois()) == 1
    survivor = merged.rois()[0]
    assert survivor["x"].start == 0.0, "the higher-confidence box survives"
    assert survivor.model_extra is not None
    assert survivor.model_extra["confidence"] == 0.9

    kept = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.6)
    ).detect(_fixed_boxes_factory(boxes))
    assert len(kept.rois()) == 2


def test_without_a_score_column_the_bigger_box_wins():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = {
        "x_min": [0, 0],
        "x_max": [10, 12],
        "y_min": [0, 0],
        "y_max": [10, 12],
    }

    table = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.5)
    ).detect(_fixed_boxes_factory(boxes))
    assert len(table.rois()) == 1
    assert table.rois()[0]["x"].length == 12.0


def test_extra_columns_ride_into_the_table():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = {
        "x_min": [0],
        "x_max": [10],
        "y_min": [0],
        "y_max": [10],
        "confidence": [0.7],
        "class_name": ["nucleus"],
    }

    table = ObjectDetectionIterator(image, axes_order="yx").detect(
        _fixed_boxes_factory(boxes)
    )
    frame = table.dataframe
    assert frame["confidence"].tolist() == [0.7]
    assert frame["class_name"].tolist() == ["nucleus"]


def test_time_frames_are_never_merged_across_t():
    """A per-frame detector's identical boxes are distinct objects per frame."""
    data = np.zeros((2, 64, 64), dtype="uint8")
    data[:, 10:20, 10:20] = 255
    _, image = _image(data, axes_names="tyx")
    iterator = ObjectDetectionIterator(image, axes_order="yx").grid(size_t=1)

    table = iterator.detect(_bright_box_detector)

    rois = table.rois()
    assert len(rois) == 2, "one object per time point, not merged across t"
    t_starts = sorted(roi["t"].start for roi in rois)
    assert t_starts == [0.0, 1.0]


def test_nothing_detected_gives_an_empty_table():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    table = ObjectDetectionIterator(image, axes_order="yx").detect(_bright_box_detector)
    assert table.rois() == []


def test_detection_table_round_trips_through_the_container():
    ome_zarr, image = _image(_two_blobs())
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    table = iterator.detect(_bright_box_detector)
    ome_zarr.add_table("detections", table)

    read_back = ome_zarr.get_table("detections")
    assert read_back.table_type() == "roi_table"
    assert len(read_back.rois()) == len(table.rois())  # type: ignore[attr-defined]


def test_column_contract_is_enforced():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = ObjectDetectionIterator(image, axes_order="yx")

    with pytest.raises(NgioValueError, match="must pin x and y"):
        iterator.detect(_fixed_boxes_factory({"x_min": [0], "x_max": [1]}))

    with pytest.raises(NgioValueError, match="but not both"):
        iterator.detect(
            _fixed_boxes_factory(
                {"x_min": [0], "x_max": [1], "y_min": [0], "y_max": [1], "z_min": [0]}
            )
        )

    with pytest.raises(NgioValueError, match="is below"):
        iterator.detect(
            _fixed_boxes_factory(
                {"x_min": [5], "x_max": [1], "y_min": [0], "y_max": [1]}
            )
        )


def test_detection_cap_is_enforced():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(max_detections_per_tile=2)
    )
    many = {
        "x_min": [0, 20, 40],
        "x_max": [1, 21, 41],
        "y_min": [0, 0, 0],
        "y_max": [1, 1, 1],
    }

    with pytest.raises(NgioValueError, match="max_detections_per_tile"):
        iterator.detect(_fixed_boxes_factory(many))


def test_rejects_bad_construction():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))

    with pytest.raises(NgioValueError, match="max_detections_per_tile"):
        NmsConfig(max_detections_per_tile=0)
    with pytest.raises(NgioValueError, match="iou_threshold"):
        NmsConfig(iou_threshold=0.0)
    with pytest.raises(NgioValueError, match="Halo along 'x' must be >= 0"):
        ObjectDetectionIterator(image, axes_order="yx").with_halo(x=-1)


def test_read_only_surface():
    """No setters; the halo IS allowed here — it is this iterator's read margin."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = ObjectDetectionIterator(image, axes_order="yx")

    assert iterator.build_numpy_setter(iterator.rois[0]) is None
    assert iterator.build_dask_setter(iterator.rois[0]) is None
    haloed = iterator.with_halo(x=4, y=4)
    assert haloed.halo == {"x": 4, "y": 4}


def test_func_receives_the_haloed_roi():
    """The roi argument covers exactly the patch the detector was handed."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    seen: list[tuple[float, float, int, int]] = []

    def recording(patch, roi):
        seen.append((roi["x"].start, roi["x"].length, *patch.shape[-2:]))
        return {"x_min": [], "x_max": [], "y_min": [], "y_max": []}

    iterator.detect(recording)

    # Tile (0, 0): halo clipped at the left border, 32 + 8 wide.
    assert (0.0, 40.0, 40, 40) in seen
    # Tile (0, 1): haloed on both sides where there is room.
    assert (24.0, 40.0, 40, 40) in seen


def test_reserved_columns_are_refused():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = {
        "x_min": [0],
        "x_max": [10],
        "y_min": [0],
        "y_max": [10],
        "label": [7],
    }

    with pytest.raises(NgioValueError, match="collide with the ROI fields"):
        ObjectDetectionIterator(image, axes_order="yx").detect(
            _fixed_boxes_factory(boxes)
        )
