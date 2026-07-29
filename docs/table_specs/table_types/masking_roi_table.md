---
description: "Masking ROI table: bounding boxes tied to the labels of a label image."
---

# Masking ROI tables

A masking ROI table is a specialised table type for representing Regions of Interest (ROIs) that are associated with specific labels in a label image.
Each row in a masking ROI table corresponds to a specific label in the label image.

Masking ROI tables serve several purposes, such as:

- Feature extraction from specific regions in the image.
- Masking specific regions in the image for further processing. For example, a masking ROI table could store the ROIs for specific tissues, and you would like to perform cell segmentation within each of them.

## Specifications

### V1

A masking ROI table must include the following metadata fields in the group attributes:

```json
{
    // ROI table metadata
    "type": "masking_roi_table",
    "table_version": "1",
    "region": {"path": "../labels/label_DAPI"}, // Path to the label image associated with this masking ROI table
    // Backend metadata
    "backend": "anndata", // the backend used to store the table, e.g. "anndata", "parquet", etc..
    "index_key": "label", // The default index key for the ROI table, which is used to identify each ROI.
    "index_type": "int", // Either "int" or "str"
}
```

Moreover the ROI table must include the following columns:

- `x_micrometer`, `y_micrometer`, `z_micrometer`: the top-left corner coordinates of the ROI in micrometers.
- `len_x_micrometer`, `len_y_micrometer`, `len_z_micrometer`: the size of the ROI in micrometers along each axis.
- `label`: An integer column label associated with the ROI, which corresponds to a specific label in the label image. This can also be the table index key.
- (Optional) `t_second` and `len_t_second`: the time coordinate of the ROI in seconds, and the length of the time coordinate in seconds. This is useful for multiplexing acquisitions.

Additionally, each ROI can include the following optional columns: see [ROI table](./roi_table.md).
