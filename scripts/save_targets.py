"""
braid_test 첫 16개 샘플의 target 이미지 (img * matte) 를 results/ 에 저장합니다.

Usage:
  python scripts/save_targets.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

DATASET_ROOT = Path("dataset3/braid")
RESULTS_DIR  = Path("results")
NUM_SAMPLES  = 16

img_dir    = DATASET_ROOT / "img"    / "test"
matte_dir  = DATASET_ROOT / "matte"  / "test"

stems = sorted(p.stem for p in img_dir.glob("*.png"))[:NUM_SAMPLES]

RESULTS_DIR.mkdir(exist_ok=True)

for idx, stem in enumerate(stems):
    img   = np.array(Image.open(img_dir   / f"{stem}.png").convert("RGB"),  dtype=np.float32) / 255.0
    matte = np.array(Image.open(matte_dir / f"{stem}.png").convert("L"),    dtype=np.float32) / 255.0

    target = (img * matte[..., None] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(target).save(RESULTS_DIR / f"{idx:04d}_target.png")
    print(f"[{idx:02d}] {stem} → results/{idx:04d}_target.png")

print(f"\n완료: {len(stems)}개 저장")
