"""
Hair Image → Sketch + Matte 추론.

Usage:
  python scripts/infer_hair2sketch.py \
    --checkpoint checkpoints/inversion/best.pth \
    --config     configs/inversion.yaml \
    --input      dataset/braid/img/test/sample.png \
    --output_dir outputs/hair2sketch/

  # 디렉토리 일괄 처리
  python scripts/infer_hair2sketch.py \
    --checkpoint checkpoints/inversion/best.pth \
    --config     configs/inversion.yaml \
    --input      dataset/braid/img/test/ \
    --output_dir outputs/hair2sketch/

출력: {stem}_sketch.png, {stem}_matte.png
→ infer_custom.py의 --sketch, --matte 입력으로 바로 사용 가능 (round-trip 검증)
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

from src.models.inversion_adapter import HairInversionAdapter
from src.models.vae_wrapper import VAEWrapper


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_image(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)


@torch.no_grad()
def infer(adapter, hair_image: torch.Tensor, device, dtype):
    hair = hair_image.to(device=device, dtype=dtype)
    sketch_pred, matte_pred, _ = adapter(hair)
    return sketch_pred.float().clamp(0, 1), matte_pred.float().clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--input",      required=True, help="이미지 파일 또는 디렉토리")
    parser.add_argument("--output_dir", default="outputs/hair2sketch/")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    model_id   = cfg["model"]["model_id"]
    local_only = cfg.get("local_files_only", False)
    grid_size  = cfg.get("grid_size", 16)

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print("Loading inversion adapter...")
    adapter = HairInversionAdapter(vae=vae, grid_size=grid_size)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(ckpt["adapter"])
    adapter = adapter.to(device=device, dtype=dtype).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        files = [input_path]

    print(f"{len(files)} images to process → {output_dir}")
    for fpath in files:
        stem = fpath.stem
        hair_image = load_image(fpath)
        sketch, matte = infer(adapter, hair_image, device, dtype)

        save_image(sketch, output_dir / f"{stem}_sketch.png")
        save_image(matte,  output_dir / f"{stem}_matte.png")
        print(f"  {stem}: sketch + matte saved")

    print("Done.")


if __name__ == "__main__":
    main()
