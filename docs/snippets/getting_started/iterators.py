"""Snippets for docs/getting_started/6_iterators.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/iterators.py
"""

# --8<-- [start:setup]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.iterators import ImageProcessingIterator
from ngio.utils import download_ome_zarr_dataset

download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset(
    "CardiomyocyteSmallMip", download_dir=download_dir, re_unzip=False
)
ome_zarr = open_ome_zarr_container(hcs_path / "B" / "03" / "0")
image = ome_zarr.get_image()
# --8<-- [end:setup]

# --8<-- [start:build]
# A new iterator covers the whole image as a single region
iterator = ImageProcessingIterator(input_image=image, output_image=image)
print(iterator)
# --8<-- [end:build]

# --8<-- [start:product]
# Narrow it to the regions named by a ROI table
iterator = iterator.product(ome_zarr.get_roi_table("FOV_ROI_table"))
print(iterator)
# --8<-- [end:product]

# --8<-- [start:inspect]
# The regions are plain Roi objects, so you can look before you process
for roi in iterator.rois[:2]:
    print(roi)
# --8<-- [end:inspect]

# --8<-- [start:synthetic_setup]
import numpy as np
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array

# A small synthetic image to demonstrate on: two bright blobs, one of them
# crossing the boundary between two 32x32 tiles.
rng = np.random.default_rng(0)
data = rng.poisson(10, size=(64, 64)).astype("uint16")
data[10:20, 26:38] += 200  # crosses the tile seam at x=32
data[40:50, 8:18] += 200
demo = create_ome_zarr_from_array(
    store=MemoryStore(), array=data, pixelsize=0.5, axes_names="yx", levels=1
)
demo_image = demo.get_image()
# --8<-- [end:synthetic_setup]

# --8<-- [start:mapper_demo]
from ngio import SegmentationIterator, ThreadedMapper
from skimage.measure import label as connected_components


def segment(patch: np.ndarray) -> np.ndarray:
    return connected_components(patch > 100).astype("uint32")


seg = demo.derive_label("seg")
seg_iterator = SegmentationIterator(demo_image, seg, axes_order="yx")

# Tiles on the OUTPUT's write grid, so the parallel map runs as one wave.
seg_iterator = seg_iterator.by_write_units()
seg_iterator.map(segment, mapper=ThreadedMapper("auto"))
print(f"labelled pixels: {int((seg.get_as_numpy() > 0).sum())}")
# --8<-- [end:mapper_demo]

# --8<-- [start:reduce_demo]
means = seg_iterator.reduce(lambda unit: float(unit.mean()))
print([round(mean, 1) for mean in means])
# --8<-- [end:reduce_demo]

# --8<-- [start:batched_demo]
from ngio import BatchedMapper, ImageProcessingIterator


def fake_model(batch: np.ndarray) -> np.ndarray:
    # Stands in for a neural network: one (B, y, x) input, one call per batch.
    return batch // 2


halved = demo.derive_image(store=MemoryStore())
nn_iterator = ImageProcessingIterator(demo_image, halved.get_image())

# size 24 over 64 clips the border tiles to 16 - the ragged shapes are padded
# to each batch's maximum before stacking, and sliced back before the write.
nn_iterator = nn_iterator.by_grid(size_x=24, size_y=24)
nn_iterator.map(fake_model, mapper=BatchedMapper(batch_size=4))
print(halved.get_image().get_as_numpy().mean().round(2))
# --8<-- [end:batched_demo]

# --8<-- [start:halo_demo]
from scipy.ndimage import uniform_filter


def smooth(patch: np.ndarray) -> np.ndarray:
    return uniform_filter(patch, size=5)


blurred = demo.derive_image(store=MemoryStore())
blur_iterator = ImageProcessingIterator(demo_image, blurred.get_image())

# 8 px of context per side; the function returns the grown region and the
# border is cropped off before the write - no seams, same write footprints.
blur_iterator = blur_iterator.by_write_units().with_halo(x=8, y=8)
blur_iterator.map(smooth, mapper=ThreadedMapper("auto"))
print(blurred.get_image().get_as_numpy().mean().round(2))
# --8<-- [end:halo_demo]

# --8<-- [start:stitch_demo]
stitched = demo.derive_label("stitched")
stitch_iterator = (
    SegmentationIterator(demo_image, stitched, axes_order="yx", stitch=True)
    .by_grid(size_x=32, size_y=32)
    .with_halo(x=8, y=8)
)
stitch_iterator.map(segment)

# Two objects, two ids - the seam-crossing blob was merged back into one.
print(f"objects: {sorted(int(v) for v in set(stitched.get_as_numpy().ravel()) - {0})}")
# --8<-- [end:stitch_demo]

# --8<-- [start:detect_demo]
from ngio import ObjectDetectionIterator, Roi


def find_bright_boxes(patch: np.ndarray, roi: Roi) -> dict[str, list]:
    ys, xs = np.nonzero(patch > 100)
    if not len(ys):
        return {"x_min": [], "x_max": [], "y_min": [], "y_max": [], "confidence": []}
    return {
        "x_min": [int(xs.min())],
        "x_max": [int(xs.max()) + 1],
        "y_min": [int(ys.min())],
        "y_max": [int(ys.max()) + 1],
        "confidence": [float((patch > 100).mean())],
    }


detect_iterator = (
    ObjectDetectionIterator(demo_image, axes_order="yx")
    .by_grid(size_x=32, size_y=32)
    .with_halo(x=8, y=8)
)
detections = detect_iterator.detect(find_bright_boxes)
demo.add_table("detections", detections)
print(detections.dataframe[["x_micrometer", "y_micrometer", "confidence"]])
# --8<-- [end:detect_demo]

# --8<-- [start:anchor_demo]
tile_roi = detect_iterator.rois[0]
box = Roi.from_values(slices={"x": (12, 30), "y": (4, 25)}, name=None, space="pixel")
abs_roi = tile_roi.anchor(box, pixel_size=demo_image.pixel_size)
print(abs_roi)
# --8<-- [end:anchor_demo]
