---
description: "ROI table: axis-aligned bounding boxes in the image space."
---

# ROI table

A ROI table defines regions of space which are axis-aligned bounding boxes in the image space.

ROI tables serve several purposes, such as:

- Storing information about the microscope field of view (FOV).
- Storing arbitrary regions of interest (ROIs).
- Acting as masks for other processes, such as segmentation or feature extraction.

## Specifications

### V1

A ROI table must include the following metadata fields in the group attributes:

```json
{
    // ROI table metadata
    "type": "roi_table",
    "table_version": "1",
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "FieldIndex", // The default index key for the ROI table, which is used to identify each ROI.
    "index_type": "str", // Either "int" or "str"
}
```

Moreover the ROI table must include the following columns:

- `x_micrometer`, `y_micrometer`, `z_micrometer`: the top-left corner coordinates of the ROI in micrometers.
- `len_x_micrometer`, `len_y_micrometer`, `len_z_micrometer`: the size of the ROI in micrometers along each axis.

Additionally, each ROI can include the following optional columns:

- `t_second` and `len_t_second`: the time coordinate of the ROI in seconds, and the length of the time coordinate in seconds. This is useful for multiplexing acquisitions.
- `x_micrometer_original`, `y_micrometer_original` and `z_micrometer_original` which are the original coordinates of the ROI in micrometers. These are typically used when the data is saved in different coordinates during conversion, e.g. to avoid overwriting data from overlapping ROIs.
- `translation_x`, `translation_y` and `translation_z`, which are used during registration of multiplexing acquisitions.

You can add further columns to the ROI table, but they are not exposed in the ROI table API.
