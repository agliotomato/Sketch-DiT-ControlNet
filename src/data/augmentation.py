"""
Augmentation pipeline for hair-DiT training.

All augmentations operate on the sample dict:
  {sketch, matte, target, img, filename}

Key design constraints:
  - SketchColorJitter: per-unique-color remapping (NOT global ColorJitter)
  - MatteBoundaryPerturbation: recomputes target after matte warp
  - AppearanceJitter: applied to target only, NOT sketch
"""

from __future__ import annotations

import random
from typing import Optional

import kornia.filters as KF
import kornia.geometry.transform as KGT
import kornia.morphology as KM
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .utils import soft_composite


# ---------------------------------------------------------------------------
# 1. Sketch Color Jitter
# ---------------------------------------------------------------------------

class SketchColorJitter:
    """
    Per-unique-color random hue remapping for colored sketches.

    Rationale: sketch color = strand structural identifier, NOT appearance.
    Each unique color in the sketch is remapped to a new random color.
    Spatial structure (which pixel belongs to which strand) is preserved.

    Args:
        p: probability of applying augmentation
        min_colors: skip aug if sketch has fewer unique colors (nearly blank)
    """

    def __init__(self, p: float = 0.8, min_colors: int = 3):
        self.p = p
        self.min_colors = min_colors

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]  # (3, H, W) float32 [0,1]
        sketch_aug = self._remap_colors(sketch)
        sample = {**sample, "sketch": sketch_aug}
        return sample

    def _remap_colors(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Find unique colors (quantized to 8-bit), remap each to a random new color.
        Preserves anti-aliasing / soft transitions via linear interpolation.
        """
        C, H, W = sketch.shape
        # Quantize to uint8 for unique color detection
        sketch_u8 = (sketch * 255).byte()  # (3, H, W)
        flat = sketch_u8.view(3, -1).T      # (N, 3)

        # Find unique colors
        unique_colors = torch.unique(flat, dim=0)  # (K, 3)
        if len(unique_colors) < self.min_colors:
            return sketch

        # Build color map: old → new
        color_map: dict[tuple, torch.Tensor] = {}
        for color in unique_colors:
            r, g, b = color.tolist()
            if r == 0 and g == 0 and b == 0:
                # keep black background as-is
                color_map[(r, g, b)] = torch.zeros(3)
            else:
                h = random.random()
                s = random.uniform(0.5, 1.0)
                v = random.uniform(0.5, 1.0)
                new_color = torch.tensor(self._hsv_to_rgb(h, s, v))
                color_map[(r, g, b)] = new_color

        # Apply mapping pixel-by-pixel via vectorized operation
        out = torch.zeros_like(sketch)
        for color, new_color in color_map.items():
            mask = (sketch_u8[0] == color[0]) & (sketch_u8[1] == color[1]) & (sketch_u8[2] == color[2])
            out[:, mask] = new_color.unsqueeze(1)

        return out

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
        if s == 0.0:
            return [v, v, v]
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        i %= 6
        return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i]


# ---------------------------------------------------------------------------
# 2. Sketch Thickness Jitter
# ---------------------------------------------------------------------------

class ThicknessJitter:
    """
    Randomly dilate sketch strokes by 0-2 pixels.

    Args:
        p:          probability of applying augmentation
        max_kernel: maximum dilation kernel size (default 3 → ±1px dilation)
    """

    def __init__(self, p: float = 0.5, max_kernel: int = 3):
        self.p = p
        self.max_kernel = max_kernel  # must be odd

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"].unsqueeze(0)  # (1, 3, H, W)
        k = random.choice([k for k in range(3, self.max_kernel + 1, 2)])  # 3 or max_kernel
        kernel = torch.ones(k, k, device=sketch.device)
        # Dilate: non-zero (stroke) pixels spread outward
        sketch_dilated = KM.dilation(sketch, kernel).squeeze(0)
        sample = {**sample, "sketch": sketch_dilated.clamp(0, 1)}
        return sample


# ---------------------------------------------------------------------------
# 3. Matte Boundary Perturbation
# ---------------------------------------------------------------------------

class MatteBoundaryPerturbation:
    """
    Apply small elastic deformation to the matte, then recompute target.

    Args:
        p:         probability of applying augmentation
        amplitude: max pixel displacement (default 4)
        sigma:     smoothness of displacement field (default 10)
    """

    def __init__(self, p: float = 0.3, amplitude: float = 4.0, sigma: float = 10.0):
        self.p = p
        self.amplitude = amplitude
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        matte = sample["matte"].unsqueeze(0)   # (1, 1, H, W)
        img   = sample["img"].unsqueeze(0)     # (1, 3, H, W)

        H, W = matte.shape[-2], matte.shape[-1]

        # Random displacement field
        noise = torch.randn(1, 2, H, W, device=matte.device) * self.amplitude
        # Smooth displacement field with Gaussian blur
        kernel_size = int(6 * self.sigma / H * H) | 1  # ensure odd
        kernel_size = max(kernel_size, 3)
        noise = KF.gaussian_blur2d(noise, (kernel_size, kernel_size), (self.sigma, self.sigma))

        # Build normalized displacement grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=matte.device),
            torch.linspace(-1, 1, W, device=matte.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # Scale noise to normalized coords
        disp = noise.permute(0, 2, 3, 1)  # (1, H, W, 2)
        disp[..., 0] /= (W / 2)
        disp[..., 1] /= (H / 2)
        grid_warped = (grid + disp).clamp(-1, 1)

        # Warp matte
        matte_warped = F.grid_sample(matte.float(), grid_warped, mode="bilinear", align_corners=True)
        matte_warped = matte_warped.squeeze(0).clamp(0, 1)  # (1, H, W)

        # Recompute target with warped matte
        target_warped = soft_composite(sample["img"], matte_warped)

        sample = {**sample, "matte": matte_warped, "target": target_warped}
        return sample


# ---------------------------------------------------------------------------
# 4. Appearance Jitter
# ---------------------------------------------------------------------------

class AppearanceJitter:
    """
    Randomly jitter brightness/contrast/saturation/hue of the target hair image.
    Applied ONLY to target, NOT to sketch (structure-appearance separation).

    Args:
        p:          probability of applying augmentation
        brightness: max ± brightness delta
        contrast:   max ± contrast delta
        saturation: max ± saturation delta
        hue:        max ± hue delta
    """

    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        target = sample["target"]  # (3, H, W) [0,1]
        matte  = sample["matte"]   # (1, H, W) [0,1]

        # Sample random factors
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        s = 1.0 + random.uniform(-self.saturation, self.saturation)
        h = random.uniform(-self.hue, self.hue)

        # Apply only to hair region (avoid changing black background)
        # Temporarily extract hair pixels, jitter, recompose
        target_jittered = TF.adjust_brightness(target, b)
        target_jittered = TF.adjust_contrast(target_jittered, c)
        target_jittered = TF.adjust_saturation(target_jittered, s)
        target_jittered = TF.adjust_hue(target_jittered, h)
        target_jittered = target_jittered.clamp(0, 1)

        # Reapply matte mask (keep background black)
        target_jittered = soft_composite(target_jittered, matte)

        sample = {**sample, "target": target_jittered}
        return sample


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class ComposeAug:
    """Sequential composition of augmentations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation_pipeline(phase: str = "pretrain") -> ComposeAug:
    """
    Build the augmentation pipeline for a given training phase.

    Args:
        phase: "pretrain" (unbraid) or "finetune" (braid)

    Returns:
        ComposeAug instance
    """
    if phase == "pretrain":
        return ComposeAug([
            SketchColorJitter(p=0.8),
            ThicknessJitter(p=0.5),
            MatteBoundaryPerturbation(p=0.3),
            AppearanceJitter(p=0.5),
        ])
    elif phase == "finetune":
        # Reduced augmentation intensity to preserve braid structure correspondence
        return ComposeAug([
            SketchColorJitter(p=0.5),
            ThicknessJitter(p=0.3),
            MatteBoundaryPerturbation(p=0.2),
            AppearanceJitter(p=0.4),
        ])
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'pretrain' or 'finetune'.")
