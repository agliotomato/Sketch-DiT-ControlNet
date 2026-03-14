"""
Inference: (sketch + matte) → hair region image using HairControlNet + SD3.5.

Saves [sketch | matte | generated | target] 4-panel grid for visual inspection.

Usage:
  python scripts/infer.py \
    --config  configs/phase2_braid.yaml \
    --checkpoint checkpoints/phase2_braid/best.pth \
    --split   braid_test \
    --num_samples 16 \
    --num_steps   20 \
    --output_dir  outputs/infer_braid
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

from src.data.dataset import HairRegionDataset
from src.models.controlnet_sd35 import HairControlNet
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    controlnet: HairControlNet,
    transformer: SD3Transformer2DModel,
    vae: VAEWrapper,
    scheduler: FlowMatchEulerDiscreteScheduler,
    sketch: torch.Tensor,   # (1, 3, 512, 512) [0,1]
    matte: torch.Tensor,    # (1, 1, 512, 512) [0,1]
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Flow matching reverse sampling. Returns (1, 3, 512, 512) in [0, 1]."""
    scheduler.set_timesteps(num_steps, device=device)

    latents = torch.randn(1, 16, 64, 64, device=device, dtype=torch.bfloat16)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="steps", leave=False)):
        sigma = scheduler.sigmas[i].to(device)
        sigmas_1d = sigma.view(1).to(dtype=torch.bfloat16)

        block_samples, null_enc_hs, null_pooled = controlnet(
            noisy_latent=latents,
            sketch=sketch.to(device=device, dtype=torch.bfloat16),
            matte=matte.to(device=device, dtype=torch.bfloat16),
            sigmas=sigmas_1d,
        )
        block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
        null_enc_hs   = null_enc_hs.to(dtype=torch.bfloat16)
        null_pooled   = null_pooled.to(dtype=torch.bfloat16)

        v_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=sigmas_1d,
            block_controlnet_hidden_states=block_samples,
            return_dict=False,
        )[0]

        latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

    image = vae.decode(latents)                     # (1, 3, 512, 512) in [-1, 1]
    image = (image.float().clamp(-1, 1) + 1) / 2   # [0, 1]
    return image


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) [0,1] tensor → (H, W, 3) uint8."""
    t = t.squeeze(0).float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def make_panel(sketch, matte, gen, target) -> np.ndarray:
    """Concatenate 4 images horizontally into one row."""
    return np.concatenate([
        to_uint8(sketch),
        to_uint8(matte),
        to_uint8(gen),
        to_uint8(target),
    ], axis=1)  # (512, 2048, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="braid_test")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--output_dir",  default="outputs/infer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading ControlNet + checkpoint...")
    controlnet = HairControlNet(
        model_id=model_id,
        vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_files_only,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    controlnet.load_state_dict(ckpt["controlnet"])
    controlnet = controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    print(f"Dataset: {args.split}")
    dataset = HairRegionDataset(split=args.split)
    n = min(args.num_samples, len(dataset))

    rows = []
    for idx in tqdm(range(n), desc="Generating"):
        data   = dataset[idx]
        sketch = data["sketch"].unsqueeze(0)
        matte  = data["matte"].unsqueeze(0)
        target = data["target"].unsqueeze(0)

        gen = run_sampling(
            controlnet=controlnet,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            sketch=sketch,
            matte=matte,
            num_steps=args.num_steps,
            device=device,
        )

        # Save individual
        Image.fromarray(to_uint8(gen.cpu())).save(output_dir / f"{idx:04d}_gen.png")

        rows.append(make_panel(sketch, matte, gen.cpu(), target))

    # Save grid (header label via blank row would need PIL draw, skip for simplicity)
    grid = np.concatenate(rows, axis=0)
    grid_path = output_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"\nGrid saved: {grid_path}")
    print(f"Columns: sketch | matte | generated | target")


if __name__ == "__main__":
    main()
