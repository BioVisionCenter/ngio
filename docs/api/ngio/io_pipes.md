# ngio.io_pipes API reference

The four io pipes and their supporting types. The deprecated pre-1.1 shells
(`NumpyRoiGetter`, `NumpyGetterMasked`, …) still import but are omitted here;
see the changelog for their replacements.

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
