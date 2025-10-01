from ngio.common._zoom import (
    InterpolationOrder,
)
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes._zoom_transform import BaseZoomTransform


class ZoomTransform(BaseZoomTransform):
    def __init__(
        self,
        input_image: AbstractImage,
        target_image: AbstractImage,
        order: InterpolationOrder = "nearest",
    ) -> None:
        super().__init__(
            input_dimensions=input_image.dimensions,
            target_dimensions=target_image.dimensions,
            order=order,
        )
