"""
Inverse Head 추론: Hair Image → Sketch + Matte.

HairToSketchDiT (DiT feature extraction + FPN mask decoder) 로드 후
hair 이미지에서 sketch_pred + matte_pred를 저장한다.

Usage:
  python scripts/infer_inverse_head.py \
    --checkpoint checkpoints/inverse_head/best.pth \
    --config     configs/inverse_head.yaml \
    --input      dataset/braid/img/test/sample.png \
    --output_dir outputs/inverse_head/

  # 디렉토리 일괄 처리
  python scripts/infer_inverse_head.py \
    --checkpoint checkpoints/inverse_head/best.pth \
    --config     configs/inverse_head.yaml \
    --input      dataset/braid/img/test/ \
    --output_dir outputs/inverse_head/

출력: {stem}_sketch.png, {stem}_matte.png
→ scripts/infer.py 또는 scripts/infer_custom.py의 --sketch, --matte 입력으로 연결 (round-trip 검증)
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

from src.models.inverse_head import HairToSketchDiT
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(
    model: HairToSketchDiT,
    hair_image: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sketch_pred, matte_pred) each (1, *, 512, 512) float in [0, 1]."""
    hair = hair_image.to(device=device, dtype=dtype)
    sketch_pred, matte_pred, _ = model(hair)
    return sketch_pred.float().clamp(0, 1), matte_pred.float().clamp(0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoints/inverse_head/best.pth")
    parser.add_argument("--config",     required=True, help="configs/inverse_head.yaml")
    parser.add_argument("--input",      required=True, help="이미지 파일 또는 디렉토리")
    parser.add_argument("--output_dir", default="outputs/inverse_head/")
    parser.add_argument("--size",       type=int, default=512)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype      = torch.bfloat16
    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)
    lora_cfg   = cfg.get("lora", {})
    grid_size  = cfg.get("grid_size", 16)

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

    # null conditioning: zero tensors (model weights carry the real null embeddings via LoRA)
    null_enc_hs = torch.zeros(1, 333, 4096, dtype=dtype, device=device)
    null_pooled = torch.zeros(1, 2048,      dtype=dtype, device=device)

    print("Building HairToSketchDiT...")
    model = HairToSketchDiT(
        transformer=transformer,
        vae=vae,
        null_enc_hs=null_enc_hs,
        null_pooled=null_pooled,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 8.0),
        grid_size=grid_size,
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

    print(f"{len(files)} image(s) → {output_dir}")
    for fpath in files:
        stem       = fpath.stem
        hair_image = load_image(fpath, size=args.size)
        sketch, matte = infer(model, hair_image, device, dtype)

        save_image(sketch, output_dir / f"{stem}_sketch.png")
        save_image(matte,  output_dir / f"{stem}_matte.png")
        print(f"  {stem}: sketch + matte saved")

    print("Done.")


if __name__ == "__main__":
    main()
