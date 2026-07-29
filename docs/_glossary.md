*[OME-Zarr]: Cloud-optimised file format for bioimaging data, built on Zarr and the OME-NGFF specification
*[NGFF]: Next Generation File Format — the OME specification that OME-Zarr implements
*[ROI]: Region of Interest — a rectangular region of an image, defined in world or pixel coordinates
*[ROIs]: Regions of Interest — rectangular regions of an image, defined in world or pixel coordinates
*[FOV]: Field of View — the area captured in a single microscope acquisition
*[FOVs]: Fields of View — the areas captured in single microscope acquisitions
*[HCS]: High-Content Screening — plate-based imaging where each well holds one or more images
*[masking ROI]: A ROI table indexed by label id, mapping each segmented object to its bounding region
*[pyramid level]: One resolution of the multiscale image pyramid; level 0 is the highest resolution
*[acquisition]: One imaging round of a plate; a well may contain images from several acquisitions
*[backend]: The on-disk serialisation used for a table (anndata, parquet, csv or json)
*[index_key]: The table column used as the row index when a table is written to disk
*[consolidate]: Propagate a write made at one pyramid level to all the other levels
*[derive]: Create a new image or label that inherits metadata and geometry from an existing one
