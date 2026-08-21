# ngio.io_pipes API reference

The four io pipes and their supporting types. The deprecated pre-1.1 shells
still import until `ngio=1.2` but are omitted here; their replacements are
the bare pipes: `*RoiGetter`/`*RoiSetter` become `NumpyGetter`/`NumpySetter`
(or the dask pair) with `roi=`, and `*GetterMasked`/`*SetterMasked` become
the same pipes with `MaskTransform` in `transforms=` (reads) or
`merge=MaskMerge(...)` (writes).

::: ngio.io_pipes
    options:
      members:
        - NumpyGetter
        - NumpySetter
        - DaskGetter
        - DaskSetter
        - DataGetter
        - DataSetter
        - DataGetterProtocol
        - DataSetterProtocol
        - TransformProtocol
        - IoPipeContext
        - MergePolicy
        - MergeRule
        - MergeInput
        - SlicingOps
        - AxesOps
        - ChunkRect
