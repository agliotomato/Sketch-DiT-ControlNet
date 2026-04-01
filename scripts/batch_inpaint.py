"""
Batch inpainting: results/composite/ 전체에 boundary inpainting 적용.

모델은 한 번만 로드하고 전체 샘플에 재사용한다.

Usage:
  python scripts/batch_inpaint.py
  python scripts/batch_inpaint.py --max_id 2676 --dilate 20 --strength 0.45
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inpaint_boundary import build_boundary_mask, tensor_to_pil, load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite_dir", default="results/composite")
    parser.add_argument("--matte_dir",     default="dataset/braid/matte/test")
    parser.add_argument("--output_dir",    default="results/final")
    parser.add_argument("--max_id",    type=int,   default=2676)
    parser.add_argument("--dilate",    type=int,   default=20)
    parser.add_argument("--strength",  type=float, default=0.45)
    parser.add_argument("--steps",     type=int,   default=20)
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    composite_dir = Path(args.composite_dir)
    matte_dir     = Path(args.matte_dir)
    output_dir    = Path(args.output_dir)

    files = sorted(composite_dir.glob("braid_*.png"))
    files = [f for f in files if int(f.stem.split("_")[1]) <= args.max_id]
    print(f"Found {len(files)} files (up to braid_{args.max_id})")

    # 모델 한 번만 로드
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    for comp_path in files:
        stem = comp_path.stem  # e.g. braid_2534
        matte_path  = matte_dir  / f"{stem}.png"
        output_path = output_dir / f"{stem}.png"

        if not matte_path.exists():
            print(f"  [SKIP] matte not found: {matte_path}")
            continue

        composited = load_image(str(comp_path), "RGB")
        matte      = load_image(str(matte_path), "L")
        matte_bin  = (matte > 0.5).float()

        mask = build_boundary_mask(matte, args.dilate)

        composited_pil = tensor_to_pil(composited)
        mask_pil       = tensor_to_pil(mask)

        result_pil = pipe(
            prompt="natural hair, photorealistic, seamless blending",
            image=composited_pil,
            mask_image=mask_pil,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
        ).images[0]

        result_t = transforms.ToTensor()(result_pil).unsqueeze(0)
        final = result_t * (1 - matte_bin) + composited * matte_bin
        final = final.clamp(0, 1)

        output_dir.mkdir(parents=True, exist_ok=True)
        final_pil = tensor_to_pil(final)
        final_pil.save(str(output_path))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
