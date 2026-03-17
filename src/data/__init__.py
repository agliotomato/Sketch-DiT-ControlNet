from .dataset import HairRegionDataset
from .augmentation import (
    StrokeColorSampler,
    ThicknessJitter,
    MatteBoundaryPerturbation,
    build_augmentation_pipeline,
)
from .utils import soft_composite, resize_matte_to_latent

__all__ = [
    "HairRegionDataset",
    "StrokeColorSampler",
    "ThicknessJitter",
    "MatteBoundaryPerturbation",
    "build_augmentation_pipeline",
    "soft_composite",
    "resize_matte_to_latent",
]
