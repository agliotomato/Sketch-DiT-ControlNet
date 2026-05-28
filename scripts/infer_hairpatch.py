"""
Sketch + Matte → HairControlNet + DiT → Hairpatch (단일 이미지 출력)

stage3 ControlNet checkpoint으로 sketch/matte를 hair 이미지로 생성.

Usage:
  # 디렉토리 일괄 처리 (파일명 매칭)
  python scripts/infer_hairpatch.py \
    --sketch     dataset/braid/sketch/test/ \
    --matte      dataset/braid/matte/test/ \
    --output_dir outputs/hairpatch/

  # 단일 파일
  python scripts/infer_hairpatch.py \
    --sketch     path/to/sketch.png \
    --matte      path/to/matte.png \
    --output_dir outputs/hairpatch/

출력: {stem}_hairpatch.png
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

from src.models.controlnet_sd35 import HairControlNet
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
# Image I/O
# ---------------------------------------------------------------------------

_to_rgb  = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
_to_gray = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def load_sketch(path: Path) -> torch.Tensor:
    return _to_rgb(Image.open(path).convert("RGB")).unsqueeze(0)


def load_matte(path: Path) -> torch.Tensor:
    return _to_gray(Image.open(path).convert("L")).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    controlnet: HairControlNet,
    transformer: SD3Transformer2DModel,
    vae: VAEWrapper,
    scheduler: FlowMatchEulerDiscreteScheduler,
    sketch: torch.Tensor,
    matte: torch.Tensor,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(1, 16, 64, 64, device=device, dtype=dtype)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="  steps", leave=False)):
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
    return (image.float().clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def collect_pairs(sketch_arg: str, matte_arg: str | None) -> list[tuple[Path, Path | None, str]]:
    sketch_path = Path(sketch_arg)
    if sketch_path.is_dir():
        files = sorted(sketch_path.glob("*.png")) + sorted(sketch_path.glob("*.jpg"))
        pairs = []
        for sf in files:
            mf = None
            if matte_arg:
                md = Path(matte_arg)
                for ext in (".png", ".jpg"):
                    cand = md / (sf.stem + ext)
                    if cand.exists():
                        mf = cand
                        break
            pairs.append((sf, mf, sf.stem))
        return pairs
    else:
        mf = Path(matte_arg) if matte_arg else None
        return [(sketch_path, mf, sketch_path.stem)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch",       default="dataset/braid/sketch/test/")
    parser.add_argument("--matte",        default="dataset/braid/matte/test/")
    parser.add_argument("--forward_ckpt", default="checkpoints/inverse_stage3/last_controlnet.pth")
    parser.add_argument("--config",       default="configs/inverse_stage3.yaml")
    parser.add_argument("--num_steps",    type=int, default=20)
    parser.add_argument("--output_dir",   default="outputs/hairpatch/")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print("Loading DiT...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=dtype, local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    print(f"Loading ControlNet: {args.forward_ckpt}")
    controlnet = HairControlNet(
        model_id=model_id,
        vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_only,
    )
    ckpt = torch.load(args.forward_ckpt, map_location="cpu", weights_only=True)
    controlnet.load_state_dict(ckpt["controlnet"])
    controlnet = controlnet.to(device=device, dtype=dtype).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_only,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.sketch, args.matte)
    print(f"\n{len(pairs)}개 이미지 처리 → {output_dir}\n")

    for sketch_path, matte_path, stem in pairs:
        sketch = load_sketch(sketch_path)

        if matte_path and matte_path.exists():
            matte = load_matte(matte_path)
        else:
            if matte_path:
                print(f"  [{stem}] matte 없음: {matte_path} → sketch 전경으로 대체")
            matte = (sketch.mean(dim=1, keepdim=True) > 0.05).float()

        print(f"  {stem}: 샘플링 중 ({args.num_steps} steps)...")
        hair_recon = run_sampling(
            controlnet, transformer, vae, scheduler,
            sketch, matte, args.num_steps, device, dtype,
        )

        out_path = output_dir / f"{stem}_hairpatch.png"
        to_pil(hair_recon.cpu()).save(out_path)
        print(f"  {stem}: 저장 → {out_path}")

    print(f"\n완료: {output_dir}")


if __name__ == "__main__":
    main()
