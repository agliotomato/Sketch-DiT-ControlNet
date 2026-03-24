"""
HairSalon(GAN) vs Ours(DiT) 비교 grid 생성.

패널 구성 (4열):
  Sketch | HairSalon (GAN) | Ours (DiT) | Target

파일명 규칙:
  - DiT 결과:      results/{idx:04d}_gen.png
  - Target:        results/{idx:04d}_target.png
  - HairSalon:     checkpoints/hairsalon_results/{style}_test/{stem}.png
  - Sketch:        dataset3/{style}/sketch/test/{stem}.png

Usage:
  python scripts/compare_hairsalon.py --style braid
  python scripts/compare_hairsalon.py --style braid --max_samples 16

전제:
  run_hairsalon_baseline.py 를 먼저 실행해서
  checkpoints/hairsalon_results/{style}_test/ 가 존재해야 함
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import HairRegionDataset

PANEL = 512
HEADER_H = 28
COLS = ["Sketch", "HairSalon (GAN)", "Ours (DiT)", "Target"]


def resize(img: np.ndarray, size: int = PANEL) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def make_header() -> np.ndarray:
    w = PANEL * len(COLS)
    img = Image.new("RGB", (w, HEADER_H), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for i, name in enumerate(COLS):
        draw.text((i * PANEL + PANEL // 2, HEADER_H // 2), name,
                  fill=(210, 210, 210), font=font, anchor="mm")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style",       choices=["braid", "unbraid"], default="braid")
    parser.add_argument("--our_dir",     default="results")
    parser.add_argument("--output_dir",  default="compare")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    our_dir       = Path(args.our_dir)
    hairsalon_dir = Path("checkpoints/hairsalon_results") / f"{args.style}_test"
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hairsalon_dir.exists():
        print(f"[ERROR] HairSalon 결과 없음: {hairsalon_dir}")
        print("먼저 실행: python scripts/run_hairsalon_baseline.py --style", args.style)
        return

    dataset = HairRegionDataset(split=f"{args.style}_test")
    stems   = dataset.stems
    if args.max_samples > 0:
        stems = stems[:args.max_samples]

    rows  = []
    n_ok  = 0
    skips = []

    for idx, stem in enumerate(tqdm(stems, desc="Comparing")):
        hs_path     = hairsalon_dir / f"{stem}.png"
        our_path    = our_dir / f"{idx:04d}_gen.png"
        target_path = our_dir / f"{idx:04d}_target.png"

        if not hs_path.exists():
            skips.append(f"{stem}: hairsalon 결과 없음")
            continue
        if not our_path.exists():
            skips.append(f"{idx:04d}: DiT 결과 없음")
            continue

        sketch_np = resize(np.array(Image.open(
            Path("dataset3") / args.style / "sketch" / "test" / f"{stem}.png"
        ).convert("RGB")))
        hs_np     = resize(np.array(Image.open(hs_path).convert("RGB")))
        our_np    = resize(np.array(Image.open(our_path).convert("RGB")))

        if target_path.exists():
            target_np = resize(np.array(Image.open(target_path).convert("RGB")))
        else:
            target_np = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)

        row = np.concatenate([sketch_np, hs_np, our_np, target_np], axis=1)
        rows.append(row)
        n_ok += 1

        Image.fromarray(row).save(output_dir / f"{idx:04d}_{stem}_compare.png")

    if skips:
        print(f"\n[WARNING] {len(skips)} 샘플 스킵:")
        for s in skips[:5]:
            print(f"  {s}")

    if not rows:
        print("비교할 샘플이 없습니다.")
        return

    header = make_header()
    grid   = np.concatenate([header] + rows, axis=0)
    grid_path = output_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)

    print(f"\n완료: {n_ok}개 샘플")
    print(f"Grid: {grid_path}")
    print(f"열:   {' | '.join(COLS)}")


if __name__ == "__main__":
    main()
