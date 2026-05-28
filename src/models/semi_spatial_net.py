"""
SemiSpatialNet: Hair Image → Colored Sketch via Dual-Path Encoder.

Fixes vs HairToSketchDiT (32×32 DiT-only, binary mask + grid color):
  - CNN spatial path: 512→256→128→64px with skip connections
    → strand-level spatial detail preserved end-to-end
  - Direct 3-ch RGB output (no binary mask + patch_color_sample pipeline)
  - Cross-attention fuses DiT semantic prior into CNN bottleneck

Architecture:
  [hair(3) + matte(1)] (B, 4, 512, 512)
    ├─ CNNEncoder
    │    L1: (B, 64,  512, 512)  ← skip
    │    L2: (B, 128, 256, 256)  ← skip
    │    L3: (B, 256, 128, 128)  ← skip
    │    L4: (B, 512,  64,  64)  ← Q for cross-attention
    │
    └─ DiTFeatureExtractor (existing, SD3.5 DiT)
         → 'late' features (B, 1536, 32, 32) ← K, V for cross-attention

  CrossAttentionFusion  (Q=CNN, K/V=DiT):
    Q  = pool(L4, 32×32) + proj → (B, 1024, d_model)
    K,V = flatten(late) + proj  → (B, 1024, d_model)
    → MHA → reshape → upsample → (B, 512, 64, 64)
    fused = L4 + attn_out

  UNetDecoder:
    64→128: cat(fused, L3=256ch) → 256ch
    128→256: cat(256,  L2=128ch) → 128ch
    256→512: cat(128,  L1= 64ch) →  64ch
    → sketch_pred (B, 3, 512, 512)  [Sigmoid]
    → stroke_mask (B, 1, 512, 512)  [Sigmoid, auxiliary BCE/Dice target]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel

from src.models.dit_feature_extractor import DiTFeatureExtractor
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_gn_silu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    g = next(g for g in [32, 16, 8, 4, 2, 1] if out_ch % g == 0)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(),
    )


def _up2(x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    if target is not None:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# CNN Encoder (pixel space)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    4-level pixel-space encoder. Input: hair(3) concatenated with matte(1) = 4ch.
    Each level adds a second conv (ResNet-lite) to improve representational capacity.
    """

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.l1a = _conv_gn_silu(in_ch, 64,  stride=1)  # 512×512
        self.l1b = _conv_gn_silu(64,    64,  stride=1)

        self.l2a = _conv_gn_silu(64,    128, stride=2)  # 256×256
        self.l2b = _conv_gn_silu(128,   128, stride=1)

        self.l3a = _conv_gn_silu(128,   256, stride=2)  # 128×128
        self.l3b = _conv_gn_silu(256,   256, stride=1)

        self.l4a = _conv_gn_silu(256,   512, stride=2)  # 64×64
        self.l4b = _conv_gn_silu(512,   512, stride=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.l1b(self.l1a(x))   # (B, 64,  512, 512)
        s2 = self.l2b(self.l2a(s1))  # (B, 128, 256, 256)
        s3 = self.l3b(self.l3a(s2))  # (B, 256, 128, 128)
        s4 = self.l4b(self.l4a(s3))  # (B, 512,  64,  64)
        return s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Cross-Attention Fusion  (Q=CNN, K/V=DiT)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Q = CNN L4 features (spatial precision, queries DiT for semantic enrichment)
    K,V = DiT 'late' tokens (hair semantic prior from SD3.5)

    Q is pooled to 32×32 before attention (matches DiT 32×32 token resolution)
    to keep attention matrix (1024×1024) memory-efficient.
    Output is upsampled back to L4 resolution (64×64) and added as residual.
    """

    def __init__(
        self,
        cnn_dim: int = 512,
        dit_dim: int = 1536,
        d_model: int = 512,
        nhead: int = 8,
    ):
        super().__init__()
        self.q_proj   = nn.Linear(cnn_dim, d_model)
        self.k_proj   = nn.Linear(dit_dim, d_model)
        self.v_proj   = nn.Linear(dit_dim, d_model)
        self.attn     = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.out_proj = nn.Linear(d_model, cnn_dim)

    def forward(
        self,
        s4: torch.Tensor,       # (B, cnn_dim, 64, 64)
        dit_late: torch.Tensor, # (B, dit_dim, 32, 32)
    ) -> torch.Tensor:          # (B, cnn_dim, 64, 64) — residual delta
        B, C, H, W = s4.shape

        # Pool CNN L4 to 32×32 → Q token sequence
        s4_q = F.avg_pool2d(s4, kernel_size=2)              # (B, C, 32, 32)
        q_flat  = s4_q.flatten(2).permute(0, 2, 1)          # (B, 1024, C)
        kv_flat = dit_late.flatten(2).permute(0, 2, 1)      # (B, 1024, dit_dim)

        q = self.q_proj(q_flat)    # (B, 1024, d_model)
        k = self.k_proj(kv_flat)   # (B, 1024, d_model)
        v = self.v_proj(kv_flat)   # (B, 1024, d_model)

        attn_out, _ = self.attn(q, k, v)                    # (B, 1024, d_model)

        out = self.out_proj(attn_out)                        # (B, 1024, C)
        out_map = out.permute(0, 2, 1).reshape(B, C, H // 2, W // 2)  # (B, C, 32, 32)

        # Upsample back to L4 spatial size (64×64)
        return F.interpolate(out_map, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# U-Net Decoder
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """Upsamples fused bottleneck to 512×512 with CNN encoder skip connections.

    Outputs both sketch_pred and stroke_mask.
    RGB color is learned end-to-end here via sketch_head.
    """

    def __init__(self):
        super().__init__()
        # 64→128:  cat(512 + 256) = 768 → 256
        self.up1 = _conv_gn_silu(512 + 256, 256)
        # 128→256: cat(256 + 128) = 384 → 128
        self.up2 = _conv_gn_silu(256 + 128, 128)
        # 256→512: cat(128 +  64) = 192 → 64
        self.up3 = _conv_gn_silu(128 +  64,  64)

        # Stroke mask: WHERE strokes are (auxiliary head)
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Sketch prediction: 3-channel direct RGB prediction (main head)
        self.sketch_head = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x:  torch.Tensor,  # (B, 512,  64,  64) — fused bottleneck
        s3: torch.Tensor,  # (B, 256, 128, 128)
        s2: torch.Tensor,  # (B, 128, 256, 256)
        s1: torch.Tensor,  # (B,  64, 512, 512)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (sketch_pred, stroke_mask)
        x = self.up1(torch.cat([_up2(x, s3), s3], dim=1))  # (B, 256, 128, 128)
        x = self.up2(torch.cat([_up2(x, s2), s2], dim=1))  # (B, 128, 256, 256)
        x = self.up3(torch.cat([_up2(x, s1), s1], dim=1))  # (B,  64, 512, 512)
        return self.sketch_head(x), self.mask_head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SemiSpatialNet(nn.Module):
    """
    Full inverse model: Hair image → (sketch_pred, stroke_mask).

    Trainable:
      - CNNEncoder        (full pixel-space, strand-level spatial detail)
      - CrossAttentionFusion
      - UNetDecoder + output heads
      - DiTFeatureExtractor: inverse LoRA (blocks 0-11) + back_blocks (12-23)

    Frozen:
      - VAE encoder
      - SD3.5 DiT base weights

    Forward signature is identical to HairToSketchDiT for trainer compatibility:
      model(hair_image, matte) → (sketch_pred, stroke_mask)
    """

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        vae: VAEWrapper,
        null_enc_hs: torch.Tensor,  # (1, 333, 4096)
        null_pooled: torch.Tensor,  # (1, 2048)
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        d_model: int = 512,
        nhead: int = 8,
    ):
        super().__init__()
        self._vae = vae  # frozen, not registered as submodule

        self.cnn_encoder = CNNEncoder(in_ch=4)  # hair(3) + matte(1)

        self.feature_extractor = DiTFeatureExtractor(
            transformer=transformer,
            null_enc_hs=null_enc_hs,
            null_pooled=null_pooled,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        dit_dim = transformer.inner_dim  # 1536 for SD3.5-medium
        self.cross_attn = CrossAttentionFusion(
            cnn_dim=512, dit_dim=dit_dim, d_model=d_model, nhead=nhead,
        )
        self.decoder = UNetDecoder()

    def forward(
        self,
        hair_image: torch.Tensor,  # (B, 3, 512, 512) in [0, 1]
        matte: torch.Tensor,       # (B, 1, 512, 512) in [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sketch_pred: (B, 3, 512, 512) in [0, 1]
            stroke_mask: (B, 1, 512, 512) in [0, 1]  — auxiliary head
        """
        dtype  = next(self.cnn_encoder.parameters()).dtype
        device = next(self.cnn_encoder.parameters()).device

        hair  = hair_image.to(device=device, dtype=dtype)
        mat   = matte.to(device=device, dtype=dtype)

        # 1. CNN Encoder: pixel space → multi-scale spatial features
        cnn_in = torch.cat([hair, mat], dim=1)        # (B, 4, 512, 512)
        s1, s2, s3, s4 = self.cnn_encoder(cnn_in)

        # 2. DiT: latent space → semantic tokens
        with torch.no_grad():
            hair_latent = self._vae.encode(hair)       # (B, 16, 64, 64)
        dit_feats = self.feature_extractor(hair_latent)  # {'early','mid','late'}

        # 3. Cross-Attention: CNN spatial queries DiT semantic tokens
        attn_delta = self.cross_attn(s4, dit_feats["late"])  # (B, 512, 64, 64)
        fused = s4 + attn_delta                               # residual fusion

        # 4. U-Net Decoder: predict sketch and stroke mask
        sketch_pred, stroke_mask = self.decoder(fused, s3, s2, s1)  # sketch_pred: (B, 3, 512, 512), stroke_mask: (B, 1, 512, 512)

        # 5. Mask outputs: sketch only at stroke locations within hair region
        stroke_mask = stroke_mask * mat
        sketch_pred = sketch_pred * stroke_mask

        return sketch_pred, stroke_mask


# ---------------------------------------------------------------------------
# Shape verification (run directly to confirm tensor shapes)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=== SemiSpatialNet Shape Verification ===\n")

    device = torch.device("cpu")
    B = 2

    # --- Stub DiT and VAE (no real weights needed for shape check) ---
    from diffusers import SD3Transformer2DModel
    print("Loading SD3 transformer stub...")
    transformer = SD3Transformer2DModel(
        sample_size=64, patch_size=2, in_channels=16,
        num_layers=4, attention_head_dim=64, num_attention_heads=24,
        joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16,
    )
    for p in transformer.parameters():
        p.requires_grad_(False)

    null_enc_hs = torch.zeros(1, 333, 4096)
    null_pooled = torch.zeros(1, 2048)

    class FakeVAE:
        def encode(self, x):
            return torch.randn(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    net = SemiSpatialNet(
        transformer=transformer,
        vae=FakeVAE(),
        null_enc_hs=null_enc_hs,
        null_pooled=null_pooled,
    )
    net.eval()

    hair  = torch.randn(B, 3, 512, 512)
    matte = torch.rand(B, 1, 512, 512)

    with torch.no_grad():
        # CNN encoder
        cnn_in = torch.cat([hair, matte], dim=1)
        s1, s2, s3, s4 = net.cnn_encoder(cnn_in)
        print(f"CNN L1 (skip):      {tuple(s1.shape)}")   # (B, 64, 512, 512)
        print(f"CNN L2 (skip):      {tuple(s2.shape)}")   # (B, 128, 256, 256)
        print(f"CNN L3 (skip):      {tuple(s3.shape)}")   # (B, 256, 128, 128)
        print(f"CNN L4 (Q src):     {tuple(s4.shape)}")   # (B, 512, 64, 64)

        # DiT semantic (stub)
        hair_latent = torch.randn(B, 16, 64, 64)
        dit_feats = net.feature_extractor(hair_latent)
        print(f"DiT 'late' (K,V src): {tuple(dit_feats['late'].shape)}")  # (B, 1536, 32, 32)

        # Cross-attention
        attn_out = net.cross_attn(s4, dit_feats["late"])
        print(f"CrossAttn output:   {tuple(attn_out.shape)}")  # (B, 512, 64, 64)

        fused = s4 + attn_out
        print(f"Fused bottleneck:   {tuple(fused.shape)}")     # (B, 512, 64, 64)

        # Decoder
        sketch, mask = net.decoder(fused, s3, s2, s1)
        print(f"sketch_pred:        {tuple(sketch.shape)}")    # (B, 3, 512, 512)
        print(f"stroke_mask:        {tuple(mask.shape)}")      # (B, 1, 512, 512)

        # Full forward
        sp, sm = net(hair, matte)
        print(f"\nFull forward OK — sketch {tuple(sp.shape)}, mask {tuple(sm.shape)}")
