"""
HairInversionAdapter: Hair Image → (Sketch, Matte)

기존 HairControlNet(Sketch→Hair)을 frozen으로 두고,
역방향 adapter를 학습해 hair photo → colored sketch + matte를 예측한다.

Architecture:
  hair_image (B, 3, 512, 512)
    → frozen VAE.encode() → hair_latent (B, 16, 64, 64)
    → AdapterEncoder (trainable, ~15M) → features (B, 256, 64, 64)
    → StrokeStructureDecoder → stroke_mask (B, 1, 512, 512)   [sigmoid]
    → patch color sampling: stroke_mask * patch_color_map(hair_image) → sketch_pred (B, 3, 512, 512)
    → MatteDecoder → matte_pred (B, 1, 512, 512)              [sigmoid]

Color sampling 방식:
  hair_image를 grid_size×grid_size 패치로 avg pool → 각 패치 평균 색 → upsample back
  → stroke_mask와 곱해서 stroke 영역에만 색 부여
  학습 augmentation(StrokeColorSampler)과 유사한 local mean color 방식.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AdapterEncoder(nn.Module):
    """
    hair_latent (B, 16, 64, 64) → adapter_features (B, 256, 64, 64)
    Spatial resolution 유지 (no stride) — 64×64 latent space에서 작동.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            ResBlock(256),
        )

    def forward(self, hair_latent: torch.Tensor) -> torch.Tensor:
        return self.net(hair_latent)


class StrokeStructureDecoder(nn.Module):
    """
    adapter_features (B, 256, 64, 64) → stroke_mask (B, 1, 512, 512)
    bilinear ×8 upsample로 64→512.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mask_low = self.conv(features)  # (B, 1, 64, 64)
        return F.interpolate(mask_low, scale_factor=8, mode="bilinear", align_corners=False)  # (B, 1, 512, 512)


class MatteDecoder(nn.Module):
    """
    adapter_features (B, 256, 64, 64) → matte_pred (B, 1, 512, 512)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        matte_low = self.conv(features)  # (B, 1, 64, 64)
        return F.interpolate(matte_low, scale_factor=8, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Color sampling (non-parametric)
# ---------------------------------------------------------------------------

def patch_color_sample(
    hair_image: torch.Tensor,
    stroke_mask: torch.Tensor,
    grid_size: int = 16,
) -> torch.Tensor:
    """
    Non-parametric color sampling: hair_image의 로컬 평균 색을 stroke 영역에 부여.

    Args:
        hair_image:  (B, 3, H, W) in [0, 1]  — target (img * matte)
        stroke_mask: (B, 1, H, W) in [0, 1]  — predicted stroke 위치 (soft sigmoid output)
        grid_size:   패치 분할 크기 (default 16 → 16×16 패치, 32px×32px each)

    Returns:
        sketch_pred: (B, 3, H, W) — stroke_mask 영역에 로컬 평균 색이 칠해진 sketch
    """
    # hair_image를 grid_size×grid_size로 평균 다운샘플 → 패치 평균 색
    hair_down = F.adaptive_avg_pool2d(hair_image, (grid_size, grid_size))  # (B, 3, G, G)
    # 다시 원본 해상도로 nearest upsample (패치 경계를 유지)
    color_map = F.interpolate(hair_down, size=hair_image.shape[-2:], mode="nearest")  # (B, 3, H, W)
    # stroke_mask (soft) × color_map → stroke 영역에만 색 부여
    return color_map * stroke_mask  # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class HairInversionAdapter(nn.Module):
    """
    Hair Image → Sketch + Matte Inversion Adapter.

    Trainable: AdapterEncoder, StrokeStructureDecoder, MatteDecoder
    Frozen:    VAEWrapper (hair_image 인코딩용)

    Args:
        vae:        frozen VAEWrapper (from HairControlNet training)
        grid_size:  color sampling 패치 크기 (default 16)
    """

    def __init__(self, vae: VAEWrapper, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self._vae = vae  # frozen, not registered as submodule

        self.encoder   = AdapterEncoder()
        self.stroke_dec = StrokeStructureDecoder()
        self.matte_dec  = MatteDecoder()

    def forward(
        self,
        hair_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hair_image: (B, 3, 512, 512) in [0, 1]  — img * matte (hair region)

        Returns:
            sketch_pred: (B, 3, 512, 512) — predicted colored sketch
            matte_pred:  (B, 1, 512, 512) — predicted matte
            stroke_mask: (B, 1, 512, 512) — raw stroke structure (before color)
        """
        dtype = next(self.encoder.parameters()).dtype
        device = next(self.encoder.parameters()).device

        # 1. Encode hair image to latent space (frozen VAE)
        with torch.no_grad():
            hair_latent = self._vae.encode(hair_image.to(device=device, dtype=dtype))  # (B, 16, 64, 64)

        # 2. Adapter encoder
        features = self.encoder(hair_latent)  # (B, 256, 64, 64)

        # 3. Predict stroke structure and matte
        stroke_mask = self.stroke_dec(features)  # (B, 1, 512, 512)
        matte_pred  = self.matte_dec(features)   # (B, 1, 512, 512)

        # 4. Non-parametric color sampling
        #    matte_pred로 색 샘플 소스(hair 영역)를 마스킹, stroke_mask만 단독으로 출력 강도 제어
        #    (이중 sigmoid 곱을 피해 stroke 색의 감쇠를 방지)
        target = hair_image.to(device=device, dtype=dtype)
        hair_masked = target * matte_pred  # 색 소스: hair 영역만
        sketch_pred = patch_color_sample(hair_masked, stroke_mask, self.grid_size)  # (B, 3, 512, 512)

        return sketch_pred, matte_pred, stroke_mask
