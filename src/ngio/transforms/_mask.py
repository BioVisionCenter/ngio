from collections.abc import Sequence

from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes import SlicingInputType, TransformProtocol
from ngio.io_pipes._mask_transform import BaseMaskTransform


class MaskTransform(BaseMaskTransform):
    """Mask the data flowing through an io pipe with a label image.

    On read, pixels outside the mask are replaced with `fill_value`; on
    write, they are protected by merging the patch with the on-disk data.
    The mask is selected by the ROI of the call: `label == roi.label`, or
    `label != 0` for an unlabelled ROI, so a single instance can be reused
    across ROIs.

    Must be the last transform in the transforms list, and the only
    read-modify-write one — the pipes raise otherwise.

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
        set_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Build a mask transform from a label image.

        Args:
            label: The label image holding the mask.
            target_image: The image the mask will be applied to; required to
                rescale a label living at a different pyramid level.
            label_transforms: Extra transforms applied to the label read.
            label_slicing_dict: Per-axis overrides for the label slicing.
            axes_order: The axes order used by the data call, so the label
                is read in matching orientation.
            fill_value: Value for outside-mask pixels on read. Writes ignore
                it and keep the on-disk data instead.
            allow_rescaling: Zoom the label to the data grid (nearest) when
                the two live at different pyramid levels.
            set_transforms: Override for the chain replayed on the write
                path's read-back; the pipe binds it for you.
        """
        super().__init__(
            label_zarr_array=label.zarr_array,
            label_dimensions=label.dimensions,
            label_transforms=label_transforms,
            label_slicing_dict=label_slicing_dict,
            axes_order=axes_order,
            fill_value=fill_value,
            allow_rescaling=allow_rescaling,
            target_dimensions=None if target_image is None else target_image.dimensions,
            set_transforms=set_transforms,
        )
