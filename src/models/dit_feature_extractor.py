"""
DiT Internal Feature Extractor for Hair→Sketch Inverse Task.

Architecture:
  hair_latent (B, 16, 64, 64)
    → frozen SD3.5 DiT (sigma=0, null conditioning)
    → hooks at ALL 24 blocks
    → learnable 3×24 aggregation weights (softmax over blocks)
    → three aggregated feature maps: early / mid / late
    → each (B, inner_dim, 32, 32)

LoRA: injected into to_q/to_k/to_v of ALL 24 blocks.
      All blocks see the same OOD input (sigma=0 clean latent),
      so all need adaptation — not just 3 fixed positions.

Aggregation: three independent softmax weight vectors over 24 blocks.
  - Eliminates arbitrary hook position selection
  - Weights are interpretable after convergence (which blocks matter for hair→sketch)
  - Follows Diffusion Hyperfeatures (NeurIPS 2023) approach

JointTransformerBlock.forward() returns (encoder_hidden_states, hidden_states).
Image features = second element. Block 23 (context_pre_only) returns (None, hs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adaptation.
    Base weights are frozen; only lora_A and lora_B are trained.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base   = base
        d_in        = base.in_features
        d_out       = base.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


class DiTFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction from frozen SD3.5 DiT via learnable aggregation.

    All 24 blocks are hooked. Three learned weight vectors (softmax over 24 blocks)
    produce early / mid / late feature maps for the FPN decoder.
    After convergence, agg_weights reveals which blocks are important for hair→sketch.
    """

    NUM_BLOCKS:   int = 24
    IMAGE_TOKENS: int = 1024   # 32×32 for 512px → 64×64 latent → patch_size=2
    SPATIAL_SIZE: int = 32
    SCALE_NAMES         = ("early", "mid", "late")

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        null_enc_hs: torch.Tensor,   # (1, 333, 4096) — T5 null text embedding
        null_pooled: torch.Tensor,   # (1, 2048)       — CLIP pooled null
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
    ):
        super().__init__()
        self._transformer = transformer   # frozen, NOT registered as submodule
        self._null_enc_hs = null_enc_hs
        self._null_pooled = null_pooled
        self.inner_dim    = transformer.inner_dim   # 1536 for SD3.5-medium

        # Learnable aggregation: 3 scales × 24 blocks
        # Row i = weight vector for scale i (early=0, mid=1, late=2)
        # Initialized to zeros → uniform softmax at start
        self.agg_weights = nn.Parameter(torch.zeros(len(self.SCALE_NAMES), self.NUM_BLOCKS))

        self.lora_modules = nn.ModuleList()
        if lora_rank > 0:
            self._inject_lora(lora_rank, lora_alpha)

    def _inject_lora(self, rank: int, alpha: float) -> None:
        """Replace to_q/to_k/to_v in ALL blocks with LoRALinear wrappers."""
        for idx in range(self.NUM_BLOCKS):
            attn = self._transformer.transformer_blocks[idx].attn
            for attr in ("to_q", "to_k", "to_v"):
                linear = getattr(attn, attr, None)
                if isinstance(linear, nn.Linear):
                    lora = LoRALinear(linear, rank, alpha)
                    setattr(attn, attr, lora)
                    self.lora_modules.append(lora)

    def forward(self, hair_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hair_latent: (B, 16, 64, 64) — clean VAE latent of hair image

        Returns:
            {'early': (B, inner_dim, 32, 32),
             'mid':   (B, inner_dim, 32, 32),
             'late':  (B, inner_dim, 32, 32)}
        """
        B      = hair_latent.shape[0]
        device = hair_latent.device
        dtype  = hair_latent.dtype

        captured: dict[int, torch.Tensor] = {}
        hooks: list = []

        def make_hook(idx: int):
            def hook(module, input, output):
                img_hs = output[1]   # (B, seq_len, inner_dim)
                captured[idx] = img_hs[:, : self.IMAGE_TOKENS, :].contiguous()
            return hook

        for idx in range(self.NUM_BLOCKS):
            h = self._transformer.transformer_blocks[idx].register_forward_hook(
                make_hook(idx)
            )
            hooks.append(h)

        try:
            null_enc = self._null_enc_hs.detach().expand(B, -1, -1).to(device=device, dtype=dtype)
            null_p   = self._null_pooled.detach().expand(B, -1).to(device=device, dtype=dtype)
            sigma    = torch.zeros(B, device=device, dtype=dtype)

            self._transformer(
                hidden_states=hair_latent,
                encoder_hidden_states=null_enc,
                pooled_projections=null_p,
                timestep=sigma,
                return_dict=False,
            )
        finally:
            for h in hooks:
                h.remove()

        # Stack all block features: (num_blocks, B, IMAGE_TOKENS, inner_dim)
        all_features = torch.stack(
            [captured[i] for i in range(self.NUM_BLOCKS)], dim=0
        )

        # Learnable weighted aggregation
        weights = self.agg_weights.softmax(dim=1)   # (3, 24)

        S = self.SPATIAL_SIZE
        result: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.SCALE_NAMES):
            w   = weights[i].view(self.NUM_BLOCKS, 1, 1, 1)  # broadcast over B, T, D
            agg = (w * all_features).sum(dim=0)               # (B, IMAGE_TOKENS, inner_dim)
            result[name] = agg.reshape(B, S, S, -1).permute(0, 3, 1, 2).contiguous()

        return result

    @torch.no_grad()
    def get_block_importance(self) -> dict[str, torch.Tensor]:
        """수렴 후 어느 블록이 각 scale에 얼마나 기여하는지 반환 (시각화용)."""
        weights = self.agg_weights.softmax(dim=1)   # (3, 24)
        return {name: weights[i].cpu() for i, name in enumerate(self.SCALE_NAMES)}
