"""Object detection over tiles: boxes to world coordinates, NMS, one table."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import NmsConfig, ObjectDetectionIterator, Roi, create_ome_zarr_from_array
from ngio.common._roi import RoiSlice
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


def _box(x: float, y: float, width: float, height: float, **extras) -> Roi:
    """A patch-local pixel-space box, the shape a detector returns."""
    return Roi.from_values(
        slices={"x": (x, width), "y": (y, height)},
        name=None,
        space="pixel",
        **extras,
    )


def _bright_box_detector(patch: np.ndarray) -> list[Roi]:
    """One box around everything bright in the patch, confidence = coverage.

    Content-driven, so every tile reports what it actually saw — which is how
    padded tiles produce genuine duplicates for NMS to resolve.
    """
    ys, xs = np.nonzero(patch > 128)
    if not len(ys):
        return []
    x_min, y_min = int(xs.min()), int(ys.min())
    return [
        _box(
            x_min,
            y_min,
            int(xs.max()) + 1 - x_min,
            int(ys.max()) + 1 - y_min,
            confidence=float((patch > 128).mean()),
        )
    ]


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
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    table = iterator.detect(_bright_box_detector)
    assert table is not None

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
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    serial = iterator.detect(_bright_box_detector)
    assert serial is not None
    threaded = iterator.detect(_bright_box_detector, mapper=ThreadedMapper(4))
    assert threaded is not None
    assert threaded.dataframe.equals(serial.dataframe)


def test_boxes_land_in_world_coordinates():
    """Patch-local pixels + padded tile origin + pixel size = world coords."""
    data = np.zeros((64, 64), dtype="uint8")
    data[40:52, 36:44] = 255  # wholly inside tile (1, 1)
    _, image = _image(data, pixelsize=0.5)
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=4, y=4)
    )

    table = iterator.detect(_bright_box_detector)
    assert table is not None

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
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=16, y=16)
    )

    table = iterator.detect(_bright_box_detector)
    assert table is not None

    roi = table.rois()[0]
    assert (roi["x"].start, roi["x"].length) == (0.0, 6.0)
    assert (roi["y"].start, roi["y"].length) == (0.0, 6.0)


def _fixed_boxes_factory(boxes):
    """A detector ignoring the pixels; used to drive NMS deterministically."""

    def detector(patch: np.ndarray):
        return boxes

    return detector


def test_iou_threshold_separates_duplicates_from_neighbours():
    """IoU ≈ 0.47: suppressed at threshold 0.4, kept apart at 0.6."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = [
        _box(0, 0, 10, 10, confidence=0.9),
        _box(2, 2, 10, 10, confidence=0.5),
    ]

    merged = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.4)
    ).detect(_fixed_boxes_factory(boxes))
    assert merged is not None
    assert len(merged.rois()) == 1
    survivor = merged.rois()[0]
    assert survivor["x"].start == 0.0, "the higher-confidence box survives"
    assert survivor.model_extra is not None
    assert survivor.model_extra["confidence"] == 0.9

    kept = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.6)
    ).detect(_fixed_boxes_factory(boxes))
    assert kept is not None
    assert len(kept.rois()) == 2


def test_without_a_score_the_bigger_box_wins():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = [_box(0, 0, 10, 10), _box(0, 0, 12, 12)]

    table = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(iou_threshold=0.5)
    ).detect(_fixed_boxes_factory(boxes))
    assert table is not None
    assert len(table.rois()) == 1
    assert table.rois()[0]["x"].length == 12.0


def test_extra_fields_ride_into_the_table():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = [_box(0, 0, 10, 10, confidence=0.7, class_name="nucleus")]

    table = ObjectDetectionIterator(image, axes_order="yx").detect(
        _fixed_boxes_factory(boxes)
    )
    assert table is not None
    frame = table.dataframe
    assert frame["confidence"].tolist() == [0.7]
    assert frame["class_name"].tolist() == ["nucleus"]


def test_time_frames_are_never_merged_across_t():
    """A per-frame detector's identical boxes are distinct objects per frame."""
    data = np.zeros((2, 64, 64), dtype="uint8")
    data[:, 10:20, 10:20] = 255
    _, image = _image(data, axes_names="tyx")
    iterator = ObjectDetectionIterator(image, axes_order="yx").by_grid(size_t=1)

    table = iterator.detect(_bright_box_detector)
    assert table is not None

    rois = table.rois()
    assert len(rois) == 2, "one object per time point, not merged across t"
    t_starts = sorted(roi["t"].start for roi in rois)
    assert t_starts == [0.0, 1.0]


def test_nothing_detected_gives_an_empty_table():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    table = ObjectDetectionIterator(image, axes_order="yx").detect(_bright_box_detector)
    assert table is not None
    assert table.rois() == []


def test_detection_table_round_trips_through_the_container():
    ome_zarr, image = _image(_two_blobs())
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    table = iterator.detect(_bright_box_detector)
    assert table is not None
    ome_zarr.add_table("detections", table)

    read_back = ome_zarr.get_table("detections")
    assert read_back.table_type() == "roi_table"
    assert len(read_back.rois()) == len(table.rois())  # type: ignore[attr-defined]


def test_roi_contract_is_enforced():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = ObjectDetectionIterator(image, axes_order="yx")

    with pytest.raises(NgioValueError, match="must return a list of Roi"):
        iterator.detect(_fixed_boxes_factory({"x_min": [0], "x_max": [1]}))

    with pytest.raises(NgioValueError, match="expected a Roi"):
        iterator.detect(_fixed_boxes_factory([{"x_min": 0}]))

    world_box = Roi.from_values(slices={"x": (0, 10), "y": (0, 10)}, name=None)
    with pytest.raises(NgioValueError, match="'world' space"):
        iterator.detect(_fixed_boxes_factory([world_box]))

    no_y = Roi.from_values(
        slices={"x": (0, 10), "z": (0, 10)}, name=None, space="pixel"
    )
    with pytest.raises(NgioValueError, match="must pin x and y"):
        iterator.detect(_fixed_boxes_factory([no_y]))

    pins_t = Roi.from_values(
        slices={"t": (0, 1), "x": (0, 10), "y": (0, 10)}, name=None, space="pixel"
    )
    with pytest.raises(NgioValueError, match="may pin only x, y"):
        iterator.detect(_fixed_boxes_factory([pins_t]))

    unbounded = Roi(
        name=None,
        slices=[RoiSlice(axis_name="x"), RoiSlice(axis_name="y", start=0, length=10)],
        space="pixel",
    )
    with pytest.raises(NgioValueError, match="not fully bounded"):
        iterator.detect(_fixed_boxes_factory([unbounded]))

    with_z = Roi.from_values(
        slices={"x": (0, 10), "y": (0, 10), "z": (0, 2)}, name=None, space="pixel"
    )
    with pytest.raises(NgioValueError, match="same box dimensionality"):
        iterator.detect(_fixed_boxes_factory([_box(0, 0, 10, 10), with_z]))


def test_score_must_be_on_every_box_or_none():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    boxes = [_box(0, 0, 10, 10, confidence=0.9), _box(20, 20, 10, 10)]

    with pytest.raises(NgioValueError, match="consistent ranking"):
        ObjectDetectionIterator(image, axes_order="yx").detect(
            _fixed_boxes_factory(boxes)
        )


def test_detection_cap_is_enforced():
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = ObjectDetectionIterator(
        image, axes_order="yx", nms=NmsConfig(max_detections_per_tile=2)
    )
    many = [_box(0, 0, 1, 1), _box(20, 0, 1, 1), _box(40, 0, 1, 1)]

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


def test_iter_as_numpy_yields_the_patch_and_its_haloed_roi():
    """The custom-flow escape hatch: each pair's roi covers exactly the patch."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    iterator = (
        ObjectDetectionIterator(image, axes_order="yx")
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    seen = [
        (roi["x"].start, roi["x"].length, *patch.shape[-2:])
        for patch, roi in iterator.iter_as_numpy()
    ]

    # Tile (0, 0): halo clipped at the left border, 32 + 8 wide.
    assert (0.0, 40.0, 40, 40) in seen
    # Tile (0, 1): haloed on both sides where there is room.
    assert (24.0, 40.0, 40, 40) in seen


def test_name_and_label_are_refused():
    """The iterator renumbers survivors itself; a preset id must not vanish."""
    _, image = _image(np.zeros((64, 64), dtype="uint8"))
    labeled = Roi.from_values(
        slices={"x": (0, 10), "y": (0, 10)}, name=None, label=7, space="pixel"
    )
    named = Roi.from_values(
        slices={"x": (0, 10), "y": (0, 10)}, name="cell", space="pixel"
    )

    for box in (labeled, named):
        with pytest.raises(NgioValueError, match="iterator itself assigns"):
            ObjectDetectionIterator(image, axes_order="yx").detect(
                _fixed_boxes_factory([box])
            )
