"""
Step 2: Composite generated hair region onto face image.

  composite = hair_region * matte + face_image * (1 - matte)

Usage:
  python scripts/composite.py \\
    --hair  path/to/hair_region.png \\
    --face  path/to/face_image.png \\
    --matte path/to/matte.png \\
    --output output/composited.png
"""

import argparse
import sys
from pathlib import Path

import torch
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


def composite(
    hair_path: str,
    face_path: str,
    matte_path: str,
    output_path: str,
    image_size: int = 512,
):
    hair  = load_image(hair_path,  "RGB", image_size)  # (1, 3, H, W) [0,1]
    face  = load_image(face_path,  "RGB", image_size)
    matte = load_image(matte_path, "L",   image_size)  # (1, 1, H, W) [0,1]

    # Simple alpha composite
    result = hair * matte + face * (1.0 - matte)
    result = result.clamp(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path)
    print(f"Composited: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hair",   required=True)
    parser.add_argument("--face",   required=True)
    parser.add_argument("--matte",  required=True)
    parser.add_argument("--output", default="output/composited.png")
    args = parser.parse_args()

    composite(args.hair, args.face, args.matte, args.output)


if __name__ == "__main__":
    main()
