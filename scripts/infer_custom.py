"""
임의 스케치/matte 파일로 hair region 추론 (학습 데이터 불필요)

학습에 쓰이지 않은 스케치(현택씨 커스텀 등) generalization 테스트용.

Usage:
  # 단일 파일
  python scripts/infer_custom.py \
    --sketch    custom/my_sketch.png \
    --matte     custom/my_matte.png \
    --checkpoint checkpoints/phase1_unbraid/best.pth \
    --config    configs/phase1_unbraid.yaml \
    --output_dir custom_results/

  # 폴더 일괄 처리 (sketch/*.png 와 matte/*.png 파일명 매칭)
  python scripts/infer_custom.py \
    --sketch    custom/sketches/ \
    --matte     custom/mattes/ \
    --checkpoint checkpoints/phase2_braid/best.pth \
    --config    configs/phase2_braid.yaml \
    --output_dir custom_results/

  # matte 없음 → 스케치 비-배경 영역을 matte로 자동 사용
  python scripts/infer_custom.py \
    --sketch    custom/my_sketch.png \
    --checkpoint checkpoints/phase2_braid/best.pth \
    --config    configs/phase2_braid.yaml \
    --output_dir custom_results/

출력:
  {name}_gen.png   — 생성된 hair patch
  {name}_panel.png — [sketch | matte | generated] 3열 비교
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
from torchvision import transforms

from src.models.controlnet_sd35 import HairControlNet
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config
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
# Image loading
# ---------------------------------------------------------------------------

_to_tensor_rgb = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
_to_tensor_gray = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def load_sketch(path: Path) -> torch.Tensor:
    """Returns (1, 3, 512, 512) float [0,1]."""
    return _to_tensor_rgb(Image.open(path).convert("RGB")).unsqueeze(0)


def load_matte(path: Path) -> torch.Tensor:
    """Returns (1, 1, 512, 512) float [0,1]."""
    return _to_tensor_gray(Image.open(path).convert("L")).unsqueeze(0)


def sketch_to_matte(sketch: torch.Tensor) -> torch.Tensor:
    """스케치 비-배경 영역(비검정)을 matte로 사용. (1,1,512,512)"""
    fg = (sketch.squeeze(0).max(dim=0).values > 0.05).float()
    return fg.unsqueeze(0).unsqueeze(0)


def recolor_sketch(sketch: torch.Tensor, hair_color: tuple[int, int, int]) -> torch.Tensor:
    """
    스케치의 모든 stroke 색을 지정한 머리 색으로 교체한다.
    배경(거의 흰색 or 거의 검정)은 그대로 유지.

    sketch: (1, 3, 512, 512) float [0,1]
    hair_color: (R, G, B) 0-255
    returns: (1, 3, 512, 512) float [0,1]
    """
    color = torch.tensor([c / 255.0 for c in hair_color], dtype=sketch.dtype)  # (3,)

    s = sketch.squeeze(0)  # (3, H, W)
    # stroke 픽셀 = 배경(흰색/검정)이 아닌 곳
    brightness = s.mean(dim=0)  # (H, W)
    is_stroke = (brightness > 0.05) & (brightness < 0.95)  # 배경 제외

    result = s.clone()
    result[:, is_stroke] = color.unsqueeze(1).expand(-1, is_stroke.sum())
    return result.unsqueeze(0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    controlnet, transformer, vae, scheduler,
    sketch, matte, num_steps, device,
) -> torch.Tensor:
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

    image = vae.decode(latents)
    return (image.float().clamp(-1, 1) + 1) / 2  # [0,1]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def to_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze(0).float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


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
                mp = Path(matte_arg)
                for ext in (".png", ".jpg"):
                    candidate = mp / (sf.stem + ext)
                    if candidate.exists():
                        mf = candidate
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
    parser.add_argument("--sketch",      required=True)
    parser.add_argument("--matte",       default=None)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--output_dir",  default="custom_results")
    parser.add_argument("--hair_color",  default=None,
                        help="stroke 색 교체 (R,G,B 0-255). 예: 139,90,43 (갈색)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id         = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    controlnet.load_state_dict(ckpt["controlnet"])
    controlnet = controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    pairs = collect_pairs(args.sketch, args.matte)
    print(f"\n{len(pairs)}개 스케치 처리\n")

    hair_color = None
    if args.hair_color:
        hair_color = tuple(int(x) for x in args.hair_color.split(","))
        print(f"Stroke 색 교체: RGB{hair_color}")

    for sketch_file, matte_file, stem in tqdm(pairs, desc="Generating"):
        sketch = load_sketch(sketch_file)

        if hair_color:
            sketch = recolor_sketch(sketch, hair_color)

        if matte_file and matte_file.exists():
            matte = load_matte(matte_file)
        else:
            if matte_file:
                print(f"  [WARNING] matte 없음: {matte_file} → 스케치 fg 영역으로 대체")
            matte = sketch_to_matte(sketch)

        gen = run_sampling(
            controlnet, transformer, vae, scheduler,
            sketch, matte, args.num_steps, device,
        )

        Image.fromarray(to_uint8(gen.cpu())).save(output_dir / f"{stem}_gen.png")

        panel = np.concatenate([to_uint8(sketch), to_uint8(matte), to_uint8(gen.cpu())], axis=1)
        Image.fromarray(panel).save(output_dir / f"{stem}_panel.png")

    print(f"\n완료: {output_dir}/")


if __name__ == "__main__":
    main()
