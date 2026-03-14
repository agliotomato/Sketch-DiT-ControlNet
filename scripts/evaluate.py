"""
정량 평가: SHR, MCS, BSS, LPIPS (matte 내부)

Usage:
  python scripts/evaluate.py \
    --config      configs/phase2_braid.yaml \
    --checkpoint  checkpoints/phase2_braid/best.pth \
    --split       braid_test \
    --num_steps   20 \
    --output_dir  eval_results/braid/
"""

import argparse
import json
import sys
from pathlib import Path

import lpips
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

from src.data.dataset import HairRegionDataset
from src.evaluation.metrics import compute_shr, compute_mcs, compute_bss
from src.models.controlnet_sd35 import HairControlNet
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Sampling (single sample)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(controlnet, transformer, vae, scheduler, sketch, matte, num_steps, device):
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(1, 16, 64, 64, device=device, dtype=torch.bfloat16)

    for i, t in enumerate(scheduler.timesteps):
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

    image = vae.decode(latents)
    image = (image.float().clamp(-1, 1) + 1) / 2  # [0, 1]
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="braid_test")
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--output_dir",  default="eval_results/braid/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading ControlNet...")
    controlnet = HairControlNet(
        model_id=model_id, vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_files_only,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    controlnet.load_state_dict(ckpt["controlnet"])
    controlnet = controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    dataset = HairRegionDataset(split=args.split)
    print(f"Dataset: {args.split} ({len(dataset)} samples)")

    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        data   = dataset[idx]
        sketch = data["sketch"].unsqueeze(0)   # (1,3,512,512) [0,1]
        matte  = data["matte"].unsqueeze(0)    # (1,1,512,512) [0,1]
        target = data["target"].unsqueeze(0)   # (1,3,512,512) [0,1]

        gen = run_sampling(
            controlnet, transformer, vae, scheduler,
            sketch, matte, args.num_steps, device,
        )  # (1,3,512,512) [0,1]

        gen_d    = gen.to(device)
        sketch_d = sketch.to(device)
        matte_d  = matte.to(device)
        target_d = target.to(device)

        shr = compute_shr(gen_d, sketch_d, matte_d).item()
        mcs = compute_mcs(gen_d, matte_d).item()
        bss = compute_bss(gen_d, sketch_d, matte_d).item()

        # LPIPS in matte region (expects [-1,1])
        gen_11    = gen_d    * 2 - 1
        target_11 = target_d * 2 - 1
        lp = lpips_fn(gen_11 * matte_d, target_11 * matte_d).item()

        results.append({
            "idx":      idx,
            "filename": data["filename"],
            "shr":      shr,
            "mcs":      mcs,
            "bss":      bss,
            "lpips":    lp,
        })

    # Summary
    n = len(results)
    avg = {k: sum(r[k] for r in results) / n for k in ("shr", "mcs", "bss", "lpips")}

    print("\n========== 정량 평가 결과 ==========")
    print(f"  SHR  (↑): {avg['shr']:.4f}   (sketch 구조 추종)")
    print(f"  MCS  (↑): {avg['mcs']:.4f}   (matte 경계 준수)")
    print(f"  BSS  (↓): {avg['bss']:.4f}   (braid 교차 구조)")
    print(f"  LPIPS(↓): {avg['lpips']:.4f}   (perceptual quality)")
    print("=====================================")

    # Save JSON
    report = {"summary": avg, "per_sample": results}
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
