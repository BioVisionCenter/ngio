# Changelog

## [Unreleased]

### Features

- add support for time in rois and roi-tables

### API Changes

- change `table_name` keyword argument to `name` for consistency in all table concatenation functions, e.g. `concatenate_image_tables`, `concatenate_image_tables_as`, etc.
- change to `Dimension` class. `get_shape` and `get_canonical_shape` have been removed, `get` uses new keyword arguments `default` instead of `strict`.
- Image like objects now have a more clean API to load data. Instead of `get_array` and `set_array`, they now use `get_as_numpy`, `get_as_dask`, and `get_as_delayed` for delayed arrays.
- Same for `get_roi` now specific methods are available:
  - for ROI objects, the `get_roi_as_numpy`, `get_roi_as_dask`, and `get_roi_as_delayed` methods
- for Image objects, the `get_*` methods now have a new `channel_name` keyword argument to specify the channel to load not by index but by name.

### Table specs

- add `t_second` and `len_t_second` to ROI tables and masking ROI tables

### Bug Fixes

- improve type consistency and remove non-necessary "type: ignore"

## [v0.3.4]

- allow to write as `anndata_v1` for backward compatibility with older ngio versions.

## [v0.3.3]

### Chores

- improve dataset download process and streamline the CI workflows

## [v0.3.2]

### API Changes

- change table backend default to `anndata_v1` for backward compatibility. This will be chaanged again when ngio `v0.2.x` is no longer supported.

### Bug Fixes

- fix [#13](https://github.com/BioVisionCenter/fractal-converters-tools/issues/13) (converters tools)
- fix [#88](https://github.com/BioVisionCenter/ngio/issues/88)
