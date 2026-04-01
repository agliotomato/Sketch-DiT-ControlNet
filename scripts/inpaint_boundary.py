"""
Step 3: Seamless boundary inpainting.

Composited image의 hair/face 경계만 SD2 inpainting으로 자연스럽게 blending한다.
Hair interior는 건드리지 않음 (originality 보존).

Mask 설계:
  - dilated = dilate(matte > 0.5, dilate_px)  # matte 바깥으로 확장
  - eroded  = erode(matte > 0.5, dilate_px)   # matte 안쪽으로 축소
  - mask    = dilated - eroded                 # 경계 안팎 ring (양쪽 모두 inpaint)

Usage:
  python scripts/inpaint_boundary.py \\
    --composited output/composited.png \\
    --matte      path/to/matte.png \\
    --output     output/final.png

  # 파라미터 실험
  python scripts/inpaint_boundary.py \\
    --composited output/composited.png \\
    --matte      path/to/matte.png \\
    --output     output/final.png \\
    --dilate 20 --strength 0.45 --steps 20
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


def load_image(path: str, mode: str = "RGB", size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)  # (1, C, H, W) [0,1]


def build_boundary_mask(matte: torch.Tensor, dilate_px: int) -> torch.Tensor:
    """
    matte: (1, 1, H, W) [0,1]
    returns mask: (1, 1, H, W) {0,1} — matte 경계 안팎을 모두 덮는 ring
    """
    matte_bin = (matte > 0.5).float()

    pad = dilate_px
    kernel = 2 * pad + 1

    # 경계 바깥: max pooling으로 dilate
    dilated = F.max_pool2d(matte_bin, kernel_size=kernel, stride=1, padding=pad)

    # 경계 안쪽: -max_pool(-x)로 erode
    eroded = 1.0 - F.max_pool2d(1.0 - matte_bin, kernel_size=kernel, stride=1, padding=pad)

    # 경계 ring = dilate - erode (안팎 모두 포함)
    mask = (dilated - eroded).clamp(0, 1)
    return mask


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, C, H, W) [0,1] → PIL RGB or L"""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def inpaint_boundary(
    composited_path: str,
    matte_path: str,
    output_path: str,
    dilate_px: int = 20,
    strength: float = 0.45,
    steps: int = 20,
    device: str = "cuda",
):
    from diffusers import StableDiffusionInpaintPipeline

    composited = load_image(composited_path, "RGB")  # (1, 3, H, W)
    matte      = load_image(matte_path,      "L")    # (1, 1, H, W)
    matte_bin  = (matte > 0.5).float()

    mask = build_boundary_mask(matte, dilate_px)     # (1, 1, H, W)

    composited_pil = tensor_to_pil(composited)
    mask_pil       = tensor_to_pil(mask)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    result_pil = pipe(
        prompt="natural hair, photorealistic, seamless blending",
        image=composited_pil,
        mask_image=mask_pil,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]

    # Hair interior는 원본 composite로 복원 (originality 보존)
    result_t = transforms.ToTensor()(result_pil).unsqueeze(0)  # (1, 3, H, W)
    hair_mask = matte_bin  # (1, 1, H, W)
    final = result_t * (1 - hair_mask) + composited * hair_mask
    final = final.clamp(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_pil = tensor_to_pil(final)
    final_pil.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composited", required=True,       help="composite.py 결과 이미지")
    parser.add_argument("--matte",      required=True,       help="soft alpha matte PNG")
    parser.add_argument("--output",     default="output/final.png")
    parser.add_argument("--dilate",     type=int,   default=20,   help="경계 ring 두께 (px)")
    parser.add_argument("--strength",   type=float, default=0.45, help="inpainting strength")
    parser.add_argument("--steps",      type=int,   default=20)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    inpaint_boundary(
        args.composited,
        args.matte,
        args.output,
        dilate_px=args.dilate,
        strength=args.strength,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
