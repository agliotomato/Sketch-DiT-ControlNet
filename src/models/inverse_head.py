"""
HairToSketchDiT: Hair Image → (Sketch, Matte) via DiT internal features.

Architecture:
  hair_image (B, 3, 512, 512)
    → frozen VAE encoder → hair_latent (B, 16, 64, 64)
    → DiTFeatureExtractor → {early, mid, late} (B, 1536, 32, 32)
        blocks 0-11:  frozen base + LoRA (to_q/k/v rank-8)
        blocks 12-23: fully unfrozen (lower LR in optimizer param group)
    → FPNMaskDecoder (trainable) → stroke_mask (B, 1, 512, 512)
    → FPNMaskDecoder (trainable) → matte_pred  (B, 1, 512, 512)
    → patch_color_sample(hair * matte_pred, stroke_mask) → sketch_pred

No image generation decoder. No VAE decoder on output.
FPN decoder uses skip connections from all three feature scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import SD3Transformer2DModel

from src.models.inversion_adapter import patch_color_sample
from src.models.vae_wrapper import VAEWrapper
from src.models.dit_feature_extractor import DiTFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_gn_silu(in_ch: int, out_ch: int) -> nn.Sequential:
    g = next(g for g in [32, 16, 8, 4, 2, 1] if out_ch % g == 0)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(),
    )


def _up2(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# FPN Mask Decoder
# ---------------------------------------------------------------------------

class FPNMaskDecoder(nn.Module):
    """
    UNet-style FPN decoder: DiT multi-scale features → binary mask.

    All DiT features are at 32×32 (flat transformer token space).
    Injection schedule (pseudo-FPN by injecting earlier features at higher scales):
      f_late  (semantic)  → start at 32×32
      f_mid   (flow)      → inject at 64×64   (f_mid upsampled ×2)
      f_early (texture)   → inject at 128×128  (f_early upsampled ×4)
      continue plain upsample: 128 → 256 → 512
    """

    def __init__(self, hidden_dim: int, out_channels: int = 1):
        super().__init__()
        H = hidden_dim

        # Project DiT hidden_dim → working channels per scale
        self.proj_late  = _conv_gn_silu(H, 256)   # semantic
        self.proj_mid   = _conv_gn_silu(H, 128)   # flow
        self.proj_early = _conv_gn_silu(H,  64)   # texture

        # Decoder stages
        self.stage_32  = _conv_gn_silu(256,       256)   # 32×32
        self.merge_64  = _conv_gn_silu(256 + 128, 128)   # 64×64,  cat mid
        self.merge_128 = _conv_gn_silu(128 +  64,  64)   # 128×128, cat early
        self.stage_256 = _conv_gn_silu( 64,        32)   # 256×256
        self.stage_512 = _conv_gn_silu( 32,        16)   # 512×512

        self.out = nn.Sequential(
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {'early', 'mid', 'late'} each (B, hidden_dim, 32, 32) float32
        Returns:
            mask: (B, out_channels, 512, 512) in [0, 1]
        """
        f_e = self.proj_early(features["early"].float())   # (B,  64, 32, 32)
        f_m = self.proj_mid(features["mid"].float())       # (B, 128, 32, 32)
        f_l = self.proj_late(features["late"].float())     # (B, 256, 32, 32)

        # 32×32 — semantic processing
        x = self.stage_32(f_l)                                        # (B, 256, 32, 32)

        # 32→64 — inject mid-level features
        x = self.merge_64(torch.cat([_up2(x), _up2(f_m)], dim=1))    # (B, 128, 64, 64)

        # 64→128 — inject early features (upsampled ×4 from 32)
        f_e_128 = F.interpolate(f_e, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.merge_128(torch.cat([_up2(x), f_e_128], dim=1))     # (B, 64, 128, 128)

        # 128→256→512
        x = self.stage_256(_up2(x))                                   # (B, 32, 256, 256)
        x = self.stage_512(_up2(x))                                   # (B, 16, 512, 512)

        return self.out(x)                                             # (B, out_ch, 512, 512)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HairToSketchDiT(nn.Module):
    """
    Full inverse model: Hair image → Sketch + Matte.

    Trainable:
      - DiTFeatureExtractor.lora_modules  (LoRA on 3 DiT blocks' attention)
      - stroke_decoder (FPNMaskDecoder)
      - matte_decoder  (FPNMaskDecoder)

    Frozen:
      - VAE encoder
      - SD3.5 DiT (base weights, only LoRA deltas train)
    """

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        vae: VAEWrapper,
        null_enc_hs: torch.Tensor,   # (1, 333, 4096) from frozen ControlNet
        null_pooled: torch.Tensor,   # (1, 2048)
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        grid_size: int = 16,
    ):
        super().__init__()
        self._vae       = vae          # frozen, not registered
        self.grid_size  = grid_size

        self.feature_extractor = DiTFeatureExtractor(
            transformer=transformer,
            null_enc_hs=null_enc_hs,
            null_pooled=null_pooled,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        hidden_dim = transformer.inner_dim   # 1536 for SD3.5-medium
        self.stroke_decoder = FPNMaskDecoder(hidden_dim, out_channels=1)
        self.matte_decoder  = FPNMaskDecoder(hidden_dim, out_channels=1)

    def forward(
        self,
        hair_image: torch.Tensor,   # (B, 3, 512, 512) in [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sketch_pred: (B, 3, 512, 512)
            matte_pred:  (B, 1, 512, 512)
            stroke_mask: (B, 1, 512, 512)
        """
        dtype  = next(self.stroke_decoder.parameters()).dtype
        device = next(self.stroke_decoder.parameters()).device

        hair = hair_image.to(device=device, dtype=dtype)

        # 1. Frozen VAE encoder
        with torch.no_grad():
            hair_latent = self._vae.encode(hair)          # (B, 16, 64, 64)

        # 2. DiT feature extraction (LoRA trainable, DiT base frozen)
        features = self.feature_extractor(hair_latent)    # {early, mid, late}

        # 3. Decode stroke mask and matte
        stroke_mask = self.stroke_decoder(features)       # (B, 1, 512, 512)
        matte_pred  = self.matte_decoder(features)        # (B, 1, 512, 512)

        # 4. Non-parametric color sampling (no learned color decoder)
        hair_masked = hair * matte_pred
        sketch_pred = patch_color_sample(
            hair_masked.float(), stroke_mask.float(), self.grid_size
        ).to(dtype=dtype)

        return sketch_pred, matte_pred, stroke_mask
