from collections.abc import Sequence

from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes import SlicingInputType, TransformProtocol
from ngio.io_pipes._mask_transform import BaseMaskMerge, BaseMaskTransform


def _label_kwargs(
    label: AbstractImage,
    target_image: AbstractImage | None,
    label_transforms: Sequence[TransformProtocol] | None,
    label_slicing_dict: dict[str, SlicingInputType] | None,
    axes_order: Sequence[str] | None,
    allow_rescaling: bool,
) -> dict:
    return {
        "label_zarr_array": label.zarr_array,
        "label_dimensions": label.dimensions,
        "label_transforms": label_transforms,
        "label_slicing_dict": label_slicing_dict,
        "axes_order": axes_order,
        "allow_rescaling": allow_rescaling,
        "target_dimensions": None if target_image is None else target_image.dimensions,
    }


class MaskTransform(BaseMaskTransform):
    """Read a region with everything outside a label's mask replaced.

    A read-side transform: pixels outside the mask come back as `fill_value`.
    The mask is chosen by the ROI of the call — `label == roi.label`, or
    `label != 0` for an unlabelled ROI — so one instance serves every ROI.

    To *write* a region without disturbing the pixels outside the mask, use
    `MaskMerge` in the `merge=` slot. That is a different operation — it
    depends on what is already on disk — and putting this transform on a write
    raises rather than silently doing nothing.

    Example:
        ```python
        mask = MaskTransform(label=nuclei, target_image=image)
        patch = image.get_roi(roi, transforms=[mask])
        ```
    """

    def __init__(
        self,
        label: AbstractImage,
        target_image: AbstractImage | None = None,
        *,
        label_transforms: Sequence[TransformProtocol] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        axes_order: Sequence[str] | None = None,
        fill_value: int | float = 0,
        allow_rescaling: bool = True,
    ) -> None:
        """Build a mask transform from a label image.

        Args:
            label: The label image holding the mask.
            target_image: The image the mask will be applied to; required to
                rescale a label living at a different pyramid level.
            label_transforms: Extra transforms applied to the label read.
            label_slicing_dict: Per-axis overrides for the label slicing.
            axes_order: The axes order used by the data call, so the label is
                read in matching orientation.
            fill_value: What outside-mask pixels read as.
            allow_rescaling: Zoom the label to the data grid (nearest) when the
                two live at different pyramid levels.
        """
        super().__init__(
            fill_value=fill_value,
            **_label_kwargs(
                label,
                target_image,
                label_transforms,
                label_slicing_dict,
                axes_order,
                allow_rescaling,
            ),
        )


class MaskMerge(BaseMaskMerge):
    """Write a region while keeping the pixels outside a label's mask.

    A merge policy, passed as `merge=` on a write. It runs after the transform
    chain, against the array's own contents, so the protected pixels are
    carried through byte-identically — not read back through the chain and
    re-inverted, which would resample them whenever a transform is lossy.

    The mask is chosen by the ROI of the call, exactly as for `MaskTransform`,
    so one instance serves every ROI.

    Example:
        ```python
        merge = MaskMerge(label=nuclei, target_image=image)
        image.set_roi(roi, patch, merge=merge)
        ```
    """

    def __init__(
        self,
        label: AbstractImage,
        target_image: AbstractImage | None = None,
        *,
        label_transforms: Sequence[TransformProtocol] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        axes_order: Sequence[str] | None = None,
        allow_rescaling: bool = True,
    ) -> None:
        """Build a mask merge from a label image.

        Args:
            label: The label image holding the mask.
            target_image: The image the mask will be applied to; required to
                rescale a label living at a different pyramid level.
            label_transforms: Extra transforms applied to the label read.
            label_slicing_dict: Per-axis overrides for the label slicing.
            axes_order: The axes order used by the data call, so the label is
                read in matching orientation.
            allow_rescaling: Zoom the label to the data grid (nearest) when the
                two live at different pyramid levels.
        """
        super().__init__(
            **_label_kwargs(
                label,
                target_image,
                label_transforms,
                label_slicing_dict,
                axes_order,
                allow_rescaling,
            )
        )
