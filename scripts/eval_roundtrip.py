"""
Round-trip 평가: hair → inverse (HairToSketchDiT) → sketch_pred → forward (HairControlNet) → hair_recon
→ LPIPS(hair_recon, hair_original)  ← Cyclic Supervised 핵심 지표

Usage:
  python scripts/eval_roundtrip.py \
    --inverse_ckpt   checkpoints/inverse_head/best.pth \
    --inverse_config configs/inverse_head.yaml \
    --forward_ckpt   checkpoints/phase2_braid/best.pth \
    --forward_config configs/phase2_braid.yaml \
    --split          braid_test \
    --num_samples    50 \
    --num_steps      20 \
    --output_dir     outputs/eval_roundtrip/

출력:
  - LPIPS / SSIM / PSNR 평균값 콘솔 출력
  - outputs/eval_roundtrip/{idx:04d}_panel.png
    [hair_orig | sketch_pred | matte_pred | hair_recon]
  - outputs/eval_roundtrip/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import lpips as lpips_lib
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

from src.data.dataset import HairRegionDataset
from src.models.controlnet_sd35 import HairControlNet
from src.models.inverse_head import HairToSketchDiT
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_inverse_model(cfg: dict, device, dtype) -> HairToSketchDiT:
    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)
    lora_cfg   = cfg.get("lora", {})

    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    null_enc_hs = torch.zeros(1, 333, 4096, dtype=dtype, device=device)
    null_pooled = torch.zeros(1, 2048,      dtype=dtype, device=device)

    model = HairToSketchDiT(
        transformer=transformer,
        vae=vae,
        null_enc_hs=null_enc_hs,
        null_pooled=null_pooled,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 8.0),
        grid_size=cfg.get("grid_size", 16),
    )
    return model, vae


def build_forward_model(cfg: dict, device, dtype):
    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)

    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()

    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()

    controlnet = HairControlNet(
        model_id=model_id,
        vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_only,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_only,
    )

    return vae, transformer, controlnet, scheduler


# ---------------------------------------------------------------------------
# Forward sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    controlnet: HairControlNet,
    transformer: SD3Transformer2DModel,
    vae: VAEWrapper,
    scheduler: FlowMatchEulerDiscreteScheduler,
    sketch: torch.Tensor,   # (1, 3, 512, 512)
    matte: torch.Tensor,    # (1, 1, 512, 512)
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(1, 16, 64, 64, device=device, dtype=dtype)

    for i, t in enumerate(scheduler.timesteps):
        sigma     = scheduler.sigmas[i].to(device)
        sigmas_1d = sigma.view(1).to(dtype=dtype)

        block_samples, null_enc_hs, null_pooled = controlnet(
            noisy_latent=latents,
            sketch=sketch.to(device=device, dtype=dtype),
            matte=matte.to(device=device, dtype=dtype),
            sigmas=sigmas_1d,
        )
        block_samples = [s.to(dtype=dtype) for s in block_samples]

        v_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=null_enc_hs.to(dtype=dtype),
            pooled_projections=null_pooled.to(dtype=dtype),
            timestep=sigmas_1d,
            block_controlnet_hidden_states=block_samples,
            return_dict=False,
        )[0]

        latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

    image = vae.decode(latents)
    return (image.float().clamp(-1, 1) + 1) / 2   # [0, 1]


# ---------------------------------------------------------------------------
# Metrics (no torchmetrics dependency)
# ---------------------------------------------------------------------------

def _psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def _ssim(pred: torch.Tensor, target: torch.Tensor,
          window_size: int = 11, sigma: float = 1.5) -> float:
    import torch.nn.functional as F
    pred   = pred.float()
    target = target.float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    C = pred.shape[1]
    kernel = kernel.expand(C, 1, window_size, window_size).to(pred.device)
    pad = window_size // 2
    mu1  = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2  = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sig1 = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1_sq
    sig2 = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sig12 = F.conv2d(pred  * target, kernel, padding=pad, groups=C) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def to_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze(0).float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def make_panel(*tensors) -> np.ndarray:
    return np.concatenate([to_uint8(t) for t in tensors], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse_ckpt",   required=True)
    parser.add_argument("--inverse_config", required=True)
    parser.add_argument("--forward_ckpt",   required=True)
    parser.add_argument("--forward_config", required=True)
    parser.add_argument("--split",          default="braid_test")
    parser.add_argument("--num_samples",    type=int, default=50)
    parser.add_argument("--num_steps",      type=int, default=20)
    parser.add_argument("--output_dir",     default="outputs/eval_roundtrip/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    inv_cfg = load_config(args.inverse_config)
    fwd_cfg = load_config(args.forward_config)

    # ---- Inverse model ----
    print("Building inverse model...")
    inv_model, inv_vae = build_inverse_model(inv_cfg, device, dtype)
    inv_ckpt = torch.load(args.inverse_ckpt, map_location="cpu", weights_only=True)
    inv_model.load_state_dict(inv_ckpt["model"])
    inv_model = inv_model.to(device=device, dtype=dtype).eval()

    # ---- Forward model ----
    print("Building forward model...")
    fwd_vae, transformer, controlnet, scheduler = build_forward_model(fwd_cfg, device, dtype)
    fwd_ckpt = torch.load(args.forward_ckpt, map_location="cpu", weights_only=True)
    controlnet.load_state_dict(fwd_ckpt["controlnet"])
    controlnet = controlnet.to(device=device, dtype=dtype).eval()
    transformer = transformer.eval()

    # ---- Metrics ----
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    from torch.utils.data import ConcatDataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.split.split(",")]
    datasets = [HairRegionDataset(split=s) for s in splits]
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    n       = min(args.num_samples, len(dataset))

    lpips_scores, ssim_scores, psnr_scores = [], [], []

    print(f"Evaluating {n} samples ({args.split})...")
    for idx in tqdm(range(n)):
        data       = dataset[idx]
        hair_orig  = data["img"].unsqueeze(0).to(device, dtype=torch.float32)   # (1,3,512,512)
        matte_gt   = data["matte"].unsqueeze(0)

        # Step 1: hair → sketch_pred  (matte provided externally: GT matte)
        matte_in = matte_gt.to(device=device, dtype=dtype)
        with torch.no_grad():
            sketch_pred, _ = inv_model(hair_orig.to(dtype=dtype), matte_in)
        sketch_pred = sketch_pred.float().clamp(0, 1)

        # Step 2: sketch_pred + matte_gt → hair_recon
        hair_recon = run_sampling(
            controlnet=controlnet,
            transformer=transformer,
            vae=fwd_vae,
            scheduler=scheduler,
            sketch=sketch_pred,
            matte=matte_in.float(),
            num_steps=args.num_steps,
            device=device,
            dtype=dtype,
        ).cpu()

        # Metrics (hair region only, masked by matte_gt)
        hair_orig_cpu  = hair_orig.cpu()
        matte_gt_cpu   = matte_gt.float()
        orig_masked    = hair_orig_cpu * matte_gt_cpu
        recon_masked   = hair_recon * matte_gt_cpu

        orig_11  = (orig_masked  * 2 - 1).to(device)
        recon_11 = (recon_masked * 2 - 1).to(device)

        with torch.no_grad():
            l = lpips_fn(recon_11, orig_11).item()
            s = _ssim(recon_masked.to(device), orig_masked.to(device))
            p = _psnr(recon_masked.to(device), orig_masked.to(device))

        lpips_scores.append(l)
        ssim_scores.append(s)
        psnr_scores.append(p)

        # Save panel: hair_orig | sketch_pred | matte_gt | hair_recon
        panel = make_panel(hair_orig_cpu, sketch_pred.cpu(), matte_gt_cpu, hair_recon)
        Image.fromarray(panel).save(output_dir / f"{idx:04d}_panel.png")

    # ---- Summary ----
    metrics = {
        "lpips_mean": float(np.mean(lpips_scores)),
        "lpips_std":  float(np.std(lpips_scores)),
        "ssim_mean":  float(np.mean(ssim_scores)),
        "ssim_std":   float(np.std(ssim_scores)),
        "psnr_mean":  float(np.mean(psnr_scores)),
        "psnr_std":   float(np.std(psnr_scores)),
        "n":          n,
        "split":      args.split,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Round-trip Metrics ===")
    print(f"  LPIPS : {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}  (↓ better)")
    print(f"  SSIM  : {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}  (↑ better)")
    print(f"  PSNR  : {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB  (↑ better)")
    print(f"\nPanels saved → {output_dir}")
    print(f"Metrics JSON → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
