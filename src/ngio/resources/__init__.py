"""Predefined resources for testing and demonstration purposes."""

from pathlib import Path
from typing import Literal

from ngio.resources.resource_model import LabelsInfo, SampleInfo

resources = Path(__file__).parent.resolve()

_resources = {
    "Cardiomyocyte": SampleInfo(
        img_path=resources
        / "20200812-CardiomyocyteDifferentiation14-Cycle1_B03"
        / "raw.jpg",
        labels=[
            LabelsInfo(
                name="nuclei",
                label_path=resources
                / "20200812-CardiomyocyteDifferentiation14-Cycle1_B03"
                / "nuclei.png",
                create_masking_table=False,
                ensure_unique_labels=True,
            ),
            LabelsInfo(
                name="nuclei_mask",
                label_path=resources
                / "20200812-CardiomyocyteDifferentiation14-Cycle1_B03"
                / "mask.png",
                create_masking_table=True,
                ensure_unique_labels=False,
            ),
        ],
        xy_pixelsize=0.325,
        z_spacing=1.0,
        time_spacing=1.0,
        name="Cardiomyocyte Differentiation",
        info="20200812-CardiomyocyteDifferentiation14-Cycle1_B03",
    )
}

AVAILABLE_SAMPLES = Literal["Cardiomyocyte"]


def get_sample_info(name: AVAILABLE_SAMPLES) -> SampleInfo:
    """Get a predefined resource by name."""
    image_info = _resources.get(name)
    if image_info is None:
        raise ValueError(
            f"Sample '{name}' not found. Available samples: {_resources.keys()}"
        )
    return image_info


__all__ = ["AVAILABLE_SAMPLES", "LabelsInfo", "SampleInfo"]
