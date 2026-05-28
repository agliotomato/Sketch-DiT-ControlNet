"""
Inverse Head 추론: Hair Image → Sketch (SemiSpatialNet).

색은 원본 hair 이미지에서 직접 가져옴:
  sketch_pred = hair_image × matte × stroke_mask

Usage:
  python scripts/infer_inverse_head.py \
    --checkpoint checkpoints/inverse_semi_spatial/best.pth \
    --config     configs/inverse_semi_spatial.yaml \
    --input      dataset/braid/img/test/ \
    --matte_dir  dataset/braid/matte/test/ \
    --output_dir outputs/inverse_semi_spatial/

출력: {stem}_sketch.png
→ eval_roundtrip.py 또는 forward 추론의 --sketch 입력으로 연결
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import SD3Transformer2DModel

from src.models.semi_spatial_net import SemiSpatialNet
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)   # (1, 3, H, W) in [0, 1]


def load_matte(path: Path, size: int = 512) -> torch.Tensor:
    matte = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(matte).unsqueeze(0)   # (1, 1, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(
    model: SemiSpatialNet,
    hair_image: torch.Tensor,
    matte: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    binarize_threshold: float = 0.5,
) -> torch.Tensor:
    """Returns sketch_pred (1, 3, 512, 512) float in [0, 1].
    stroke_mask is binarized to produce clean hard-brush strokes.
    """
    hair = hair_image.to(device=device, dtype=dtype)
    mat  = matte.to(device=device, dtype=dtype)

    sketch_pred, stroke_mask = model(hair, mat)
    sketch_pred = sketch_pred.float()
    stroke_mask = stroke_mask.float()

    # sketch_pred = decoder_rgb * soft_mask  →  recover decoder_rgb, apply hard mask
    stroke_hard  = (stroke_mask > binarize_threshold).float()
    decoder_rgb  = sketch_pred / stroke_mask.clamp(min=1e-6)
    return (decoder_rgb * stroke_hard).clamp(0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--input",      required=True, help="hair 이미지 파일 또는 디렉토리")
    parser.add_argument("--matte_dir",  required=True, help="GT matte 디렉토리 (또는 SHS/BiSeNet 출력)")
    parser.add_argument("--output_dir",  default="outputs/inverse_head/")
    parser.add_argument("--size",        type=int, default=512)
    parser.add_argument("--max_images",  type=int, default=None)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype      = torch.bfloat16
    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)
    lora_cfg   = cfg.get("lora", {})

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print("Loading DiT (frozen)...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    null_enc_hs = torch.zeros(1, 333, 4096, dtype=dtype, device=device)
    null_pooled = torch.zeros(1, 2048,      dtype=dtype, device=device)

    print("Building SemiSpatialNet...")
    model = SemiSpatialNet(
        transformer=transformer,
        vae=vae,
        null_enc_hs=null_enc_hs,
        null_pooled=null_pooled,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 8.0),
        d_model=cfg.get("d_model", 512),
        nhead=cfg.get("nhead", 8),
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=dtype).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        files = [input_path]

    matte_dir = Path(args.matte_dir)
    if args.max_images:
        files = files[:args.max_images]

    print(f"{len(files)} image(s) → {output_dir}")
    for fpath in files:
        stem       = fpath.stem
        matte_path = matte_dir / f"{stem}.png"
        if not matte_path.exists():
            print(f"  {stem}: matte not found ({matte_path}), skipping")
            continue

        hair_image = load_image(fpath, size=args.size)
        matte      = load_matte(matte_path, size=args.size)
        sketch     = infer(model, hair_image, matte, device, dtype)

        save_image(sketch, output_dir / f"{stem}_sketch.png")
        print(f"  {stem}: sketch saved")

    print("Done.")


if __name__ == "__main__":
    main()
