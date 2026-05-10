"""
DiT Internal Feature Extractor for Hair→Sketch Inverse Task.

Architecture:
  hair_latent (B, 16, 64, 64)
    → SD3.5 DiT (sigma=0, null conditioning)
        blocks 0-11:  frozen base + LoRA (to_q/to_k/to_v, rank-8)
        blocks 12-23: fully unfrozen (all parameters trainable)
    → hooks at ALL 24 blocks
    → learnable 3×24 aggregation weights (softmax over blocks)
    → three aggregated feature maps: early / mid / late
    → each (B, inner_dim, 32, 32)

Partial unfreeze rationale:
  Front 12 (lower-level texture): LoRA-only — preserve general features.
  Back 12  (higher-level semantic): fully unfrozen — learn hair-specific semantics.

Aggregation: three independent softmax weight vectors over 24 blocks.
  - Eliminates arbitrary hook position selection
  - Weights interpretable after convergence

JointTransformerBlock.forward() returns (encoder_hidden_states, hidden_states).
Image features = second element. Block 23 (context_pre_only) returns (None, hs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel


class DualLoRALinear(nn.Module):
    """
    Two independent LoRA adapters on one frozen base Linear: 'forward' and 'inverse'.
    Active adapter is selected by set_mode(). Only that adapter is in the compute graph,
    so gradients naturally route to the correct adapter without explicit masking.

    Forward adapter: used during forward FM steps (Stage 3).
    Inverse adapter: used during inverse supervised steps (Stage 2 & 3).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base  = base
        d_in, d_out = base.in_features, base.out_features
        self.scale = alpha / rank

        self.fwd_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.fwd_B = nn.Parameter(torch.zeros(d_out, rank))
        self.inv_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.inv_B = nn.Parameter(torch.zeros(d_out, rank))

        self._mode = "inverse"

    def set_mode(self, mode: str) -> None:
        assert mode in ("forward", "inverse"), f"Unknown LoRA mode: {mode}"
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._mode == "forward":
            return base_out + (x @ self.fwd_A.T @ self.fwd_B.T) * self.scale
        return base_out + (x @ self.inv_A.T @ self.inv_B.T) * self.scale

    def init_inverse_from_forward(self) -> None:
        self.inv_A.data.copy_(self.fwd_A.data)
        self.inv_B.data.copy_(self.fwd_B.data)


class DiTFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction from frozen SD3.5 DiT via learnable aggregation.

    All 24 blocks are hooked. Three learned weight vectors (softmax over 24 blocks)
    produce early / mid / late feature maps for the FPN decoder.
    After convergence, agg_weights reveals which blocks are important for hair→sketch.
    """

    NUM_BLOCKS:     int = 24
    NUM_FRONT_LORA: int = 12   # blocks 0-11: LoRA only; blocks 12-23: fully unfrozen
    IMAGE_TOKENS:   int = 1024   # 32×32 for 512px → 64×64 latent → patch_size=2
    SPATIAL_SIZE:   int = 32
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

        # LoRA on front 12 blocks (0-11); back 12 blocks are fully unfrozen below
        self.lora_modules = nn.ModuleList()
        if lora_rank > 0:
            self._inject_lora(lora_rank, lora_alpha)

        # Register back blocks (12-23) as submodule so their parameters appear in
        # model.parameters() and receive gradients from both supervised and LPIPS paths.
        # Trainer freezes ALL transformer params first; this selectively re-enables them.
        self.back_blocks = nn.ModuleList(
            transformer.transformer_blocks[self.NUM_FRONT_LORA:]
        )
        for p in self.back_blocks.parameters():
            p.requires_grad_(True)

    def _inject_lora(self, rank: int, alpha: float) -> None:
        """Replace to_q/to_k/to_v in front NUM_FRONT_LORA blocks with DualLoRALinear."""
        for idx in range(self.NUM_FRONT_LORA):
            attn = self._transformer.transformer_blocks[idx].attn
            for attr in ("to_q", "to_k", "to_v"):
                linear = getattr(attn, attr, None)
                if isinstance(linear, nn.Linear):
                    lora = DualLoRALinear(linear, rank, alpha)
                    setattr(attn, attr, lora)
                    self.lora_modules.append(lora)

    # ------------------------------------------------------------------
    # LoRA mode control
    # ------------------------------------------------------------------

    def set_lora_mode(self, mode: str) -> None:
        """Switch all DualLoRALinear adapters to 'forward' or 'inverse' mode."""
        for m in self.lora_modules:
            m.set_mode(mode)

    def init_inverse_from_forward(self) -> None:
        """Warm-start: copy forward LoRA weights into inverse LoRA (Stage 2 init)."""
        for m in self.lora_modules:
            m.init_inverse_from_forward()

    def freeze_forward_lora(self) -> None:
        for m in self.lora_modules:
            m.fwd_A.requires_grad_(False)
            m.fwd_B.requires_grad_(False)

    def unfreeze_forward_lora(self) -> None:
        for m in self.lora_modules:
            m.fwd_A.requires_grad_(True)
            m.fwd_B.requires_grad_(True)

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
