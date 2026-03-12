"""
HairRegionDataset

Loads (sketch, matte, target) triplets where:
  target = img * (matte / 255.0)  — hair region on black background

Supported splits:
  "unbraid_train", "unbraid_test", "braid_train", "braid_test"
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import soft_composite

DATASET_ROOT = Path(__file__).parent.parent.parent / "dataset3"


def _build_stem_index(img_dir: Path) -> list[str]:
    """Return sorted list of file stems (without extension)."""
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No PNG files found in {img_dir}")
    return stems


class HairRegionDataset(Dataset):
    """
    Each item:
        sketch:   (3, H, W)  float32 [0, 1]  — colored sketch
        matte:    (1, H, W)  float32 [0, 1]  — soft alpha matte
        target:   (3, H, W)  float32 [0, 1]  — img * matte (hair region)
        filename: str
    """

    VALID_SPLITS = ("unbraid_train", "unbraid_test", "braid_train", "braid_test")

    def __init__(
        self,
        split: str,
        image_size: int = 512,
        augmentation: Optional[Callable] = None,
        dataset_root: Optional[Path] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        style, subset = split.rsplit("_", 1)
        root = Path(dataset_root) if dataset_root else DATASET_ROOT
        base = root / style

        self.img_dir    = base / "img"    / subset
        self.sketch_dir = base / "sketch" / subset
        self.matte_dir  = base / "matte"  / subset

        for d in [self.img_dir, self.sketch_dir, self.matte_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        self.stems = _build_stem_index(self.img_dir)
        self.augmentation = augmentation
        self.image_size = image_size

        # Basic transform: PIL → tensor [0,1], resize if needed
        self._to_tensor_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),   # [0,255] → [0,1], HWC → CHW
        ])
        self._to_tensor_gray = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        img    = Image.open(self.img_dir    / f"{stem}.png").convert("RGB")
        sketch = Image.open(self.sketch_dir / f"{stem}.png").convert("RGB")
        matte  = Image.open(self.matte_dir  / f"{stem}.png").convert("L")   # grayscale

        img_t    = self._to_tensor_rgb(img)     # (3, H, W)
        sketch_t = self._to_tensor_rgb(sketch)  # (3, H, W)
        matte_t  = self._to_tensor_gray(matte)  # (1, H, W), values [0,1]

        # target = hair region (soft composite)
        target_t = soft_composite(img_t, matte_t)  # (3, H, W)

        sample = {
            "sketch":   sketch_t,
            "matte":    matte_t,
            "target":   target_t,
            "img":      img_t,      # kept for composite.py (Step 2)
            "filename": stem,
        }

        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return sample
