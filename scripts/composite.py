"""
Step 2: Composite generated hair region onto face image.

  composite = hair_region * matte + face_image * (1 - matte)

Gaussian feathering is applied to the matte edges before blending
to create a smooth transition at the boundary.

Usage:
  python scripts/composite.py \\
    --hair  path/to/hair_region.png \\
    --face  path/to/face_image.png \\
    --matte path/to/matte.png \\
    --output output/composited.png \\
    --feather 3.0 \\
    --scale 1.2 --offset_x 10 --offset_y -20
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_image(path: str, mode: str = "RGB", size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)


def transform_patch(tensor: torch.Tensor, scale: float, offset_x: int, offset_y: int) -> torch.Tensor:
    """hair patch와 matte에 scale/translation 적용. (1, C, H, W) → (1, C, H, W)"""
    if scale == 1.0 and offset_x == 0 and offset_y == 0:
        return tensor

    _, C, H, W = tensor.shape

    # affine grid: scale + translate
    # offset은 픽셀 단위, normalized coords로 변환
    tx = offset_x / (W / 2)
    ty = offset_y / (H / 2)

    theta = torch.tensor([[
        [scale, 0,     tx],
        [0,     scale, ty],
    ]], dtype=torch.float32)

    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    transformed = F.grid_sample(tensor, grid, align_corners=False, mode="bilinear", padding_mode="zeros")
    return transformed


def composite(
    hair_path: str,
    face_path: str,
    matte_path: str,
    output_path: str,
    image_size: int = 512,
    feather_sigma: float = 3.0,
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
):
    hair  = load_image(hair_path,  "RGB", image_size)  # (1, 3, H, W) [0,1]
    face  = load_image(face_path,  "RGB", image_size)
    matte = load_image(matte_path, "L",   image_size)  # (1, 1, H, W) [0,1]

    # Scale / translation 적용 (hair + matte 동일하게)
    hair  = transform_patch(hair,  scale, offset_x, offset_y)
    matte = transform_patch(matte, scale, offset_x, offset_y)

    # Gaussian feathering: matte 경계를 부드럽게
    if feather_sigma > 0:
        kernel_size = int(feather_sigma * 6) | 1  # 홀수 보장
        matte = TF.gaussian_blur(matte, kernel_size=[kernel_size, kernel_size], sigma=feather_sigma)

    result = hair * matte + face * (1.0 - matte)
    result = result.clamp(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path)
    print(f"Composited: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hair",     required=True)
    parser.add_argument("--face",     required=True)
    parser.add_argument("--matte",    required=True)
    parser.add_argument("--output",   default="output/composited.png")
    parser.add_argument("--feather",  type=float, default=3.0,  help="Gaussian feathering sigma (0=off)")
    parser.add_argument("--scale",    type=float, default=1.0,  help="hair/matte 스케일 (1.0=원본)")
    parser.add_argument("--offset_x", type=int,   default=0,    help="x축 이동 (픽셀, 양수=오른쪽)")
    parser.add_argument("--offset_y", type=int,   default=0,    help="y축 이동 (픽셀, 양수=아래쪽)")
    args = parser.parse_args()

    composite(
        args.hair, args.face, args.matte, args.output,
        feather_sigma=args.feather,
        scale=args.scale,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
    )


if __name__ == "__main__":
    main()
