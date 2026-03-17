"""
Augmentation pipeline for hair-DiT training.

All augmentations operate on the sample dict:
  {sketch, matte, target, img, filename}

Key design constraints:
  - StrokeColorSampler: per-stroke color sampling from actual target hair pixels
    (follows SketchHairSalon paper: each iteration samples a random pixel from
     the corresponding target region → color correspondence is maintained)
  - MatteBoundaryPerturbation: recomputes target after matte warp
"""

from __future__ import annotations

import random

import kornia.filters as KF
import kornia.morphology as KM
import torch
import torch.nn.functional as F

from .utils import soft_composite


# ---------------------------------------------------------------------------
# 1. Stroke Color Sampler  (SketchHairSalon 방식)
# ---------------------------------------------------------------------------

class StrokeColorSampler:
    """
    Per-stroke color sampling from actual target hair pixels.

    SketchHairSalon 논문 방식:
      각 stroke 영역에 대해, 매 iteration마다 target 이미지의 해당 위치 픽셀 중
      무작위로 1개를 샘플링하여 stroke 색으로 할당.

    효과:
      - 색 대응 유지: stroke 색 ∈ 실제 머리 색 범위 → 모델이 색 correspondence 학습
      - 데이터 증강: 같은 구조라도 매 iteration 색이 미세하게 달라져 다양성 확보

    Args:
        p:              적용 확률
        min_pixels:     stroke 영역 픽셀 수가 이 값 미만이면 해당 stroke 색 유지
        quantize_bits:  unique stroke 검출을 위한 양자화 비트 수
    """

    def __init__(self, p: float = 1.0, min_pixels: int = 10, quantize_bits: int = 5):
        self.p = p
        self.min_pixels = min_pixels
        self.shift = 8 - quantize_bits

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]  # (3, H, W) float32 [0,1]
        target = sample["target"]  # (3, H, W) float32 [0,1]  = img * matte

        sketch_aug = self._resample_colors(sketch, target)
        return {**sample, "sketch": sketch_aug}

    def _resample_colors(self, sketch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # uint8 양자화로 unique stroke 레이블 검출
        sketch_u8 = (sketch * 255).byte()  # (3, H, W)
        sketch_q  = (sketch_u8 >> self.shift) << self.shift  # quantized

        flat_q = sketch_q.view(3, -1).T  # (N, 3)
        unique_colors = torch.unique(flat_q, dim=0)  # (K, 3)

        out = sketch.clone()

        for color in unique_colors:
            r, g, b = color.tolist()

            # 검은 배경(non-hair stroke) 은 건드리지 않음
            if r == 0 and g == 0 and b == 0:
                continue

            # 이 stroke 레이블에 해당하는 픽셀 마스크
            mask = (
                (sketch_q[0] == r) &
                (sketch_q[1] == g) &
                (sketch_q[2] == b)
            )  # (H, W) bool

            # target에서 이 stroke 위치의 실제 머리 픽셀 추출
            hair_pixels = target[:, mask]  # (3, N)

            # matte 내 유효한 픽셀만 (target이 0이 아닌 것)
            valid = hair_pixels.sum(dim=0) > 0.05  # (N,)
            if valid.sum() < self.min_pixels:
                continue

            hair_pixels_valid = hair_pixels[:, valid]  # (3, M)

            # 무작위로 픽셀 1개 샘플링
            idx = random.randint(0, hair_pixels_valid.shape[1] - 1)
            sampled_color = hair_pixels_valid[:, idx]  # (3,)

            # 해당 stroke 모든 픽셀을 sampled_color로 교체
            out[:, mask] = sampled_color.unsqueeze(1)

        return out


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
        k = random.choice([k for k in range(3, self.max_kernel + 1, 2)])
        kernel = torch.ones(k, k, device=sketch.device)
        sketch_dilated = KM.dilation(sketch, kernel).squeeze(0)
        return {**sample, "sketch": sketch_dilated.clamp(0, 1)}


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

        noise = torch.randn(1, 2, H, W, device=matte.device) * self.amplitude
        kernel_size = int(6 * self.sigma / H * H) | 1
        kernel_size = max(kernel_size, 3)
        noise = KF.gaussian_blur2d(noise, (kernel_size, kernel_size), (self.sigma, self.sigma))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=matte.device),
            torch.linspace(-1, 1, W, device=matte.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        disp = noise.permute(0, 2, 3, 1)
        disp[..., 0] /= (W / 2)
        disp[..., 1] /= (H / 2)
        grid_warped = (grid + disp).clamp(-1, 1)

        matte_warped = F.grid_sample(matte.float(), grid_warped, mode="bilinear", align_corners=True)
        matte_warped = matte_warped.squeeze(0).clamp(0, 1)

        target_warped = soft_composite(sample["img"], matte_warped)
        return {**sample, "matte": matte_warped, "target": target_warped}


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

    StrokeColorSampler (p=1.0):
      매 iteration마다 target 실제 픽셀을 샘플링하여 stroke 색 할당.
      AppearanceJitter 제거: target 색을 흔들면 stroke↔target 색 대응이 깨짐.

    Args:
        phase: "pretrain" (unbraid) or "finetune" (braid)
    """
    if phase == "pretrain":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.5),
            MatteBoundaryPerturbation(p=0.3),
        ])
    elif phase == "finetune":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.3),
            MatteBoundaryPerturbation(p=0.2),
        ])
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'pretrain' or 'finetune'.")
