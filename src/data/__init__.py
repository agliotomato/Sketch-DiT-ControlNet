from .dataset import HairRegionDataset
from .augmentation import (
    SketchColorJitter,
    ThicknessJitter,
    MatteBoundaryPerturbation,
    AppearanceJitter,
    build_augmentation_pipeline,
)
from .utils import soft_composite, resize_matte_to_latent

__all__ = [
    "HairRegionDataset",
    "SketchColorJitter",
    "ThicknessJitter",
    "MatteBoundaryPerturbation",
    "AppearanceJitter",
    "build_augmentation_pipeline",
    "soft_composite",
    "resize_matte_to_latent",
]
