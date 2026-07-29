---
description: The four ngio iterators for building scalable image-processing pipelines.
---

# 6. Iterators

When building image processing pipelines it is often useful to iterate over specific regions of the image, for example to process the image in smaller tiles or to process only specific regions of interest (ROIs).

Moreover, when working with OME-Zarr Images it is often useful to set specific broadcasting rules for the iteration, for example to iterate over all z-planes or iterate over all timepoints.

ngio provides a set of `Iterator` classes that can be used for this purpose. We provide four basic iterators:

* The `SegmentationIterator` is designed to build segmentation pipelines, where an input image is processed to produce a segmentation mask. An example use case on how to use the `SegmentationIterator` can be found in the [Image Segmentation Tutorial](../tutorials/image_segmentation.md).
* The `MaskedSegmentationIterator` is similar to the `SegmentationIterator`, but it uses a masking roi table to restrict the segmentation to masks. This is useful when you want to segment only specific regions of the image, for example, segmenting cells only within a specific tissue region. An example use case on how to use the `MaskedSegmentationIterator` can be found in the [Image Segmentation Tutorial](../tutorials/image_segmentation.md).
* The `ImageProcessingIterator` is designed to build image processing pipelines, where an input image is processed to produce a new image. An example use case on how to use the `ImageProcessingIterator` can be found in the [Image Processing Tutorial](../tutorials/image_processing.md).
* The `FeatureExtractorIterator` is a read-only iterator designed to iterate over pairs of images and labels to extract features from the image based on the labels. An example use case on how to use the `FeatureExtractorIterator` can be found in the [Feature Extraction Tutorial](../tutorials/feature_extraction.md).

A set of more complete example can be found in the [Fractal Tasks Template](https://github.com/fractal-analytics-platform/fractal-tasks-template).

## Next steps

- [Image Processing tutorial](../tutorials/image_processing.md) — an iterator applied end to end.
- [Image Segmentation tutorial](../tutorials/image_segmentation.md) — segmentation and masked segmentation.
- [Iterators API reference](../api/iterators.md) — the full iterator API.
