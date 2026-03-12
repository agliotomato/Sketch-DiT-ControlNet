"""
Visualization utilities.

Generates 4-column grids: [sketch | matte | generated | target]
for qualitative result analysis.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.data.dataset import DATASET_ROOT


def load_sample_images(stem: str, split: str, image_size: int = 256) -> dict[str, torch.Tensor]:
    """Load (sketch, matte, target) for a given stem from a split."""
    style, subset = split.rsplit("_", 1)
    base = DATASET_ROOT / style

    tf_rgb  = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    tf_gray = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    sketch = tf_rgb(Image.open(base / "sketch" / subset / f"{stem}.png").convert("RGB"))
    matte  = tf_gray(Image.open(base / "matte"  / subset / f"{stem}.png").convert("L")).expand(3, -1, -1)
    img    = tf_rgb(Image.open(base / "img"     / subset / f"{stem}.png").convert("RGB"))

    # target = img * matte
    matte_01 = matte[:1]  # single channel
    target = img * matte_01

    return {"sketch": sketch, "matte": matte, "target": target}


def save_result_grid(
    stems: list[str],
    split: str,
    output_path: str | Path,
    generated_dir: str | Path | None = None,
    image_size: int = 256,
    nrow: int = 4,
):
    """
    Save a grid of [sketch | matte | generated | target] for each stem.

    Args:
        stems:         list of file stems to include
        split:         dataset split (e.g., "braid_test")
        output_path:   where to save the grid PNG
        generated_dir: directory containing generated images (stem.png);
                       if None, shows target in the "generated" column
        image_size:    size to display each image
        nrow:          number of sample rows (each row = 4 images)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    all_images = []
    for stem in stems:
        sample = load_sample_images(stem, split, image_size)

        # Generated image (or placeholder)
        if generated_dir is not None:
            gen_path = Path(generated_dir) / f"{stem}.png"
            if gen_path.exists():
                generated = tf(Image.open(gen_path).convert("RGB"))
            else:
                generated = torch.zeros(3, image_size, image_size)
        else:
            generated = sample["target"].clone()  # show target as placeholder

        row = torch.stack([sample["sketch"], sample["matte"], generated, sample["target"]])
        all_images.append(row)

    # (N, 4, C, H, W) → flatten to (N*4, C, H, W) for make_grid
    grid_input = torch.cat(all_images, dim=0)  # (N*4, C, H, W)

    # 4 columns per row (sketch, matte, gen, target)
    grid = make_grid(grid_input, nrow=4, normalize=False, padding=2)
    save_image(grid, output_path)
    print(f"Saved grid ({len(stems)} samples) to {output_path}")
