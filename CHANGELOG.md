# Changelog

## [v0.4.2]

### API Changes

- Make roi.to_slicing_dict(pixel_size) always require pixel_size argument for consistency with other roi methods.
- Make PixelSize object a Pydantic model to allow for serialization.

### Bug Fixes

- Improve robustness when rounding Rois to pixel coordinates. 

## [v0.4.1]

### Bug Fixes
- Fix bug in zoom transform when input axes contain unknown axes (e.g. virtual axes). Now unknown axes are treated as virtual axes and set to 1 in the target shape.

## [v0.4.0]

### Features

- Add Iterators for image processing pipelines
- Add support for time in rois and roi-tables
- Building masking roi tables expanded to time series data
- Add zoom transformation
- Add support for rescaling on-the-fly masks for masked images
- Big refactor of the io pipeline to support iterators and lazy loading
- Add support for customize dimension separators and compression codecs
- Simplify AxesHandler and Dataset Classes

### API Changes

- The image-like `get_*` api have been slightly changed. Now if a single int is passed as slice_kwargs, it is interpreted as a single index. So the dimension is automatically squeezed.
- Remove the `get_*_delayed` methods, now data cam only be loaded as numpy or dask array.Use the `get_as_dask` method instead, which returns a dask array that can be used with dask delayed.
- A new model for channel selection is available. Now channels can be selected by name, index or with `ChannelSelectionModel` object.
- Change `table_name` keyword argument to `name` for consistency in all table concatenation functions, e.g. `concatenate_image_tables`,  `concatenate_image_tables_as`, etc.
- Change to `Dimension` class. `get_shape` and `get_canonical_shape` have been removed, `get` uses new keyword arguments `default` instead of `strict`.
- Image like objects now have a more clean API to load data. Instead of `get_array` and `set_array`, they now use `get_as_numpy`, and `get_as_dask` for delayed arrays.
- Also for `get_roi` now specific methods are available. For ROI objects, the `get_roi_as_numpy`, and `get_roi_as_dask` methods.
- Table ops moved to `ngio.images`
- int `label` as an explicit attribute in `Roi` objects (previously only in stored in name and relying on convention)
- Slight changes to `Image` and `Label` objects. Some minor attributes have been renamed for consistency.

### Table specs

- Add `t_second` and `len_t_second` to ROI tables and masking ROI tables

## [v0.3.5]

- Remove path normalization for images in wells. While the spec requires paths to be alphanumeric, this patch removes the normalization to allow for arbitrary image paths.

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
