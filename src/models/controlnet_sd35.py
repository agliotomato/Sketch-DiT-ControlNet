"""
HairControlNet: SD3.5 ControlNet for hair region generation.

Components:
- SD3ControlNetModel (trainable, initialized from SD3.5-medium transformer)
- MatteCNN: 1ch matte → 16ch features at latent resolution (512→64)
- null_encoder_hidden_states: nn.Parameter (1, 333, 4096), learned null text embedding
- null_pooled_projections:    nn.Parameter (1, 2048), learned null pooled embedding

Forward:
  inputs: noisy_latent (B,16,64,64), sketch (B,3,512,512), matte (B,1,512,512), sigmas (B,)
  1. sketch → frozen VAE encode → sketch_latent (B,16,64,64)
  2. matte → MatteCNN → matte_feat (B,16,64,64)
  3. ctrl_cond = sketch_latent + matte_feat
  4. SD3ControlNetModel forward → block_samples
  5. return block_samples, null_encoder_hs (expanded to B), null_pooled (expanded to B)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import SD3ControlNetModel, SD3Transformer2DModel

import torch.nn.functional as F

from src.models.vae_wrapper import VAEWrapper


class MatteCNN(nn.Module):
    """
    Lightweight CNN to embed 1-channel matte into 16-channel latent-resolution features.

    Spatial: 512 → 256 → 128 → 64  (three stride-2 convolutions)
    Channels:  1  →  16 →  32 →  16
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 1 → 16, 512 → 256
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # Block 2: 16 → 32, 256 → 128
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            # Block 3: 32 → 16, 128 → 64
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )

    def forward(self, matte: torch.Tensor) -> torch.Tensor:
        """
        Args:
            matte: (B, 1, 512, 512) in [0, 1]
        Returns:
            feat: (B, 16, 64, 64)
        """
        return self.net(matte)


class SketchDecoderHead(nn.Module):
    """
    Lightweight decoder: 12 ControlNet block_samples → sketch (B, 3, 512, 512).

    block_samples shape: list of (B, seq_len, 1152)  — SD3 sequence format.
    Image tokens are the first spatial_size² entries (32×32=1024 for 512px input).
    """

    def __init__(self, num_blocks: int = 12, hidden_dim: int = 1152, image_tokens: int = 1024):
        super().__init__()
        self.spatial_size = int(image_tokens ** 0.5)  # 32
        self.num_tokens   = image_tokens

        self.block_weights = nn.Parameter(torch.zeros(num_blocks))

        # 32×32 → 512×512 via 4× bilinear upsample (32→64→128→256→512)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 1),
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, block_samples: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            block_samples: list of N × (B, seq_len, hidden_dim) tensors
        Returns:
            sketch_pred: (B, 3, 512, 512) in [0, 1]
        """
        weights = self.block_weights.softmax(dim=0)  # (N,)
        # Weighted sum over blocks; slice first image_tokens along seq dim
        agg = sum(
            w * b[:, : self.num_tokens, :]
            for w, b in zip(weights, block_samples)
        )  # (B, image_tokens, hidden_dim)

        B = agg.shape[0]
        S = self.spatial_size
        # Reshape to spatial (B, hidden_dim, S, S) — cast to float32 for conv stability
        spatial = agg.reshape(B, S, S, -1).permute(0, 3, 1, 2).to(dtype=torch.float32)
        out = self.decoder(spatial)  # (B, 3, 512, 512) float32
        return out.to(dtype=agg.dtype)


class HairControlNet(nn.Module):
    """
    SD3.5 ControlNet model for hair region synthesis.

    Trainable components:
      - controlnet (SD3ControlNetModel)
      - matte_cnn (MatteCNN)
      - null_encoder_hidden_states (nn.Parameter)
      - null_pooled_projections (nn.Parameter)

    Frozen components (passed in, not stored as submodules):
      - vae (VAEWrapper) — used only for sketch encoding in forward()
    """

    # Null embedding shapes for SD3.5-medium
    NULL_ENC_SHAPE = (1, 333, 4096)
    NULL_POOLED_SHAPE = (1, 2048)

    def __init__(
        self,
        model_id: str,
        vae: VAEWrapper,
        num_layers: int = 12,
        local_files_only: bool = False,
        use_sketch_decoder: bool = False,
    ):
        super().__init__()

        # Load transformer temporarily just to initialize ControlNet
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )

        self.controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=num_layers,
            load_weights_from_transformer=True,
            # SD3ControlNetModel defaults to extra_conditioning_channels=1,
            # so pos_embed_input expects 17ch. We provide 17ch ctrl_cond:
            #   16ch: sketch_latent + matte_feat
            #    1ch: raw matte_latent (explicit spatial mask)
        )
        # Free transformer memory — it's held separately in Trainer
        del transformer
        torch.cuda.empty_cache()

        self.matte_cnn = MatteCNN()

        self.sketch_decoder: SketchDecoderHead | None = (
            SketchDecoderHead() if use_sketch_decoder else None
        )

        # Learned null text conditioning (trained alongside ControlNet)
        self.null_encoder_hidden_states = nn.Parameter(
            torch.zeros(*self.NULL_ENC_SHAPE)
        )
        self.null_pooled_projections = nn.Parameter(
            torch.zeros(*self.NULL_POOLED_SHAPE)
        )

        # Keep VAE reference (frozen, not a submodule to avoid double registration)
        self._vae = vae

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sketch: torch.Tensor,
        matte: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_latent: (B, 16, 64, 64) noisy target latent
            sketch:       (B, 3, 512, 512) colored sketch in [0, 1]
            matte:        (B, 1, 512, 512) hair matte in [0, 1]
            sigmas:       (B,) flow matching sigma values

        Returns:
            block_samples:      list of ControlNet residuals
            null_encoder_hs:    (B, 333, 4096) expanded null text embeddings
            null_pooled:        (B, 2048) expanded null pooled embeddings
        """
        B = noisy_latent.shape[0]
        device = noisy_latent.device
        dtype = noisy_latent.dtype

        # 1. Encode sketch through frozen VAE → latent conditioning
        sketch_latent = self._vae.encode(sketch.to(dtype=dtype))   # (B, 16, 64, 64)
        sketch_latent = sketch_latent.to(device=device, dtype=dtype)

        # 2. Encode matte through trainable CNN → latent-resolution features
        matte_feat = self.matte_cnn(matte.to(device=device, dtype=dtype))  # (B, 16, 64, 64)

        # 3. Combine into control conditioning (17ch for SD3ControlNetModel API)
        #    16ch: sketch_latent + matte_feat  (structural + matte learned features)
        #     1ch: raw matte downsampled       (explicit spatial mask)
        matte_latent = F.interpolate(
            matte.to(device=device, dtype=dtype), size=(64, 64), mode="bilinear", align_corners=False
        )  # (B, 1, 64, 64)
        ctrl_cond = torch.cat([sketch_latent + matte_feat, matte_latent], dim=1)  # (B, 17, 64, 64)

        # 4. Expand null embeddings to batch size
        null_enc_hs = self.null_encoder_hidden_states.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_pooled = self.null_pooled_projections.expand(B, -1).to(device=device, dtype=dtype)

        # 5. Sigmas → timestep format expected by SD3 (1D, float)
        timestep = sigmas.to(device=device, dtype=dtype)

        # 6. ControlNet forward
        block_samples = self.controlnet(
            hidden_states=noisy_latent,
            controlnet_cond=ctrl_cond,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=timestep,
            return_dict=False,
        )[0]

        return block_samples, null_enc_hs, null_pooled

    def _get_features_impl(
        self,
        condition_image: torch.Tensor,
        matte: torch.Tensor,
        enable_grad: bool = False,
    ) -> list[torch.Tensor]:
        """
        조건 이미지 → ControlNet block_samples 추출 (공유 구현).

        condition_image: (B, 3, 512, 512) — sketch (inversion용) 또는 hair image (cycle loss용)
        matte:           (B, 1, 512, 512)
        enable_grad:     True → VAE encode without no_grad so gradients flow back to
                         condition_image (needed for cycle loss to reach inverse model params).
                         False (default) → standard no_grad encode for efficiency.
        """
        B = condition_image.shape[0]
        device = condition_image.device
        dtype = condition_image.dtype

        if enable_grad:
            # Gradient-enabled encode: allows ∂cycle_loss/∂condition_image → inverse model params
            cond_latent = self._vae.encode_for_grad(condition_image.to(dtype=dtype)).to(device=device, dtype=dtype)
        else:
            cond_latent = self._vae.encode(condition_image.to(dtype=dtype)).to(device=device, dtype=dtype)

        matte_feat   = self.matte_cnn(matte.to(device=device, dtype=dtype))
        matte_latent = F.interpolate(
            matte.to(device=device, dtype=dtype), size=(64, 64), mode="bilinear", align_corners=False
        )
        ctrl_cond = torch.cat([cond_latent + matte_feat, matte_latent], dim=1)  # (B, 17, 64, 64)

        null_enc_hs = self.null_encoder_hidden_states.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_pooled = self.null_pooled_projections.expand(B, -1).to(device=device, dtype=dtype)

        dummy_latent = torch.zeros(B, 16, 64, 64, device=device, dtype=dtype)
        dummy_sigma  = torch.zeros(B, device=device, dtype=dtype)

        return self.controlnet(
            hidden_states=dummy_latent,
            controlnet_cond=ctrl_cond,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=dummy_sigma,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def get_features(
        self,
        sketch: torch.Tensor,
        matte: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Inversion Phase B용: sketch + matte → block_samples (gradient 없음).
        Cycle self-distillation에서 gradient가 필요하면 _get_features_impl() 직접 호출.
        """
        return self._get_features_impl(sketch, matte)


# ---------------------------------------------------------------------------
# Matte-gated residual helpers
# ---------------------------------------------------------------------------

def make_matte_tok(matte: torch.Tensor) -> torch.Tensor:
    """matte (B,1,512,512) → token grid (B,1024,1), SD3 patchify와 동일 순서.

    SD3 patchify: Conv2d(patch=2) → flatten(2).transpose(1,2)
    512px → latent 64² → patch2 → 32×32 = 1024 tokens
    """
    tok = F.interpolate(matte, size=(32, 32), mode="bilinear", align_corners=False)
    return tok.flatten(2).transpose(1, 2)  # (B, 1024, 1)


def gate_block_samples(
    block_samples: list[torch.Tensor],
    matte: torch.Tensor,
    schedule: str,
    image_tokens: int = 1024,
    num_transformer_blocks: int = 24,
) -> list[torch.Tensor]:
    """schedule에 따라 block_samples의 image token 부분에 soft matte gate 적용.

    block_sample[k]는 transformer blocks 2k, 2k+1에 대응 (interval=2).
      back_only  : k=6..11 → transformer block 12..23
      front_only : k=0..5  → transformer block 0..11
      all        : 전체
    text token (seq[1024:])은 게이팅하지 않음.
    """
    if schedule == "none":
        return block_samples

    matte_tok = make_matte_tok(matte).to(
        dtype=block_samples[0].dtype, device=block_samples[0].device
    )  # (B, 1024, 1)
    interval = num_transformer_blocks // len(block_samples)  # 24 // 12 = 2
    half = num_transformer_blocks // 2                       # 12

    gated = []
    for k, res in enumerate(block_samples):
        block_idx = k * interval
        apply = {
            "front_only": block_idx < half,
            "all":        True,
            "back_only":  block_idx >= half,
        }[schedule]
        if apply:
            img = res[:, :image_tokens, :] * matte_tok  # (B, 1024, 1152)
            txt = res[:, image_tokens:, :]
            gated.append(torch.cat([img, txt], dim=1))
        else:
            gated.append(res)
    return gated
