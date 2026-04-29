"""
Hair region FID 계산 (standalone)

Usage:
  python scripts/eval_fid.py \
    --dit_dir custom_results/dit/weighted_sum \
    --out_dir eval_results/weighted_sum

실제 GT hair crop vs 생성 hair crop 을 128x128로 저장 후 FID 계산.
"""

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pytorch_fid import fid_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GT_IMG_DIR   = Path("dataset/braid/img/test")
GT_MATTE_DIR = Path("dataset/braid/matte/test")
GAN_DIR      = Path("custom_results/gan/shs")


def get_hair_crop(img: np.ndarray, matte: np.ndarray, size: int = 128) -> np.ndarray | None:
    hair = matte > 127
    if hair.sum() < 64:
        return None
    ys, xs = np.where(hair)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1].copy()
    mask_crop = hair[y0:y1, x0:x1]
    crop[~mask_crop] = 0
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (size, size))


def get_bnd_crop(img: np.ndarray, matte: np.ndarray, size: int = 128) -> np.ndarray | None:
    bnd = (matte >= 25) & (matte <= 230)
    if bnd.sum() < 16:
        return None
    ys, xs = np.where(bnd)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1].copy()
    bnd_crop = bnd[y0:y1, x0:x1]
    crop[~bnd_crop] = 0
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return cv2.resize(crop, (size, size))


def save_crops(crops: list[np.ndarray], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(crops):
        if c is not None:
            Image.fromarray(c).save(out_dir / f"{i:05d}.png")


def run_fid(real_dir: Path, fake_dir: Path) -> float:
    n_real = len(list(real_dir.glob("*.png")))
    n_fake = len(list(fake_dir.glob("*.png")))
    print(f"  real={n_real}, fake={n_fake}")
    if n_real < 2 or n_fake < 2:
        return float("nan")
    return fid_score.calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=32, device=DEVICE, dims=2048, num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_dir", required=True)
    parser.add_argument("--gan_dir", default=str(GAN_DIR))
    parser.add_argument("--out_dir", default="eval_results")
    args = parser.parse_args()

    dit_dir = Path(args.dit_dir)
    gan_dir = Path(args.gan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in GT_IMG_DIR.glob("*.png"))
    print(f"이미지: {len(stems)}개")

    real_hair, real_bnd = [], []
    gan_hair,  gan_bnd  = [], []
    dit_hair,  dit_bnd  = [], []

    for stem in tqdm(stems, desc="crop 추출"):
        gt  = np.array(Image.open(GT_IMG_DIR   / f"{stem}.png").convert("RGB"))
        matte = np.array(Image.open(GT_MATTE_DIR / f"{stem}.png").convert("L"))

        real_hair.append(get_hair_crop(gt, matte))
        real_bnd.append(get_bnd_crop(gt, matte))

        gan_path = gan_dir / f"{stem}.png"
        dit_path = dit_dir / f"{stem}_full.png"

        gan_img = np.array(Image.open(gan_path).convert("RGB")) if gan_path.exists() else None
        dit_img = np.array(Image.open(dit_path).convert("RGB")) if dit_path.exists() else None

        gan_hair.append(get_hair_crop(gan_img, matte) if gan_img is not None else None)
        gan_bnd.append(get_bnd_crop(gan_img, matte)   if gan_img is not None else None)
        dit_hair.append(get_hair_crop(dit_img, matte) if dit_img is not None else None)
        dit_bnd.append(get_bnd_crop(dit_img, matte)   if dit_img is not None else None)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_crops([x for x in real_hair if x is not None], tmp / "real_hair")
        save_crops([x for x in real_bnd  if x is not None], tmp / "real_bnd")
        save_crops([x for x in gan_hair  if x is not None], tmp / "gan_hair")
        save_crops([x for x in gan_bnd   if x is not None], tmp / "gan_bnd")
        save_crops([x for x in dit_hair  if x is not None], tmp / "dit_hair")
        save_crops([x for x in dit_bnd   if x is not None], tmp / "dit_bnd")

        print("\nHair FID 계산...")
        gan_hair_fid = run_fid(tmp / "real_hair", tmp / "gan_hair")
        dit_hair_fid = run_fid(tmp / "real_hair", tmp / "dit_hair")

        print("\nBoundary FID 계산...")
        gan_bnd_fid = run_fid(tmp / "real_bnd", tmp / "gan_bnd")
        dit_bnd_fid = run_fid(tmp / "real_bnd", tmp / "dit_bnd")

    def fmt(v):
        return f"{v:.2f}" if not (v != v) else "N/A"

    print("\n" + "=" * 46)
    print(f"{'Metric':<22}  {'GAN':>10}  {'DiT':>10}")
    print("-" * 46)
    print(f"{'Hair FID ↓':<22}  {fmt(gan_hair_fid):>10}  {fmt(dit_hair_fid):>10}")
    print(f"{'Boundary FID ↓':<22}  {fmt(gan_bnd_fid):>10}  {fmt(dit_bnd_fid):>10}")
    print("=" * 46)

    # CSV 저장
    import csv
    with open(out_dir / "fid_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "GAN", "DiT"])
        w.writerow(["Hair FID ↓", fmt(gan_hair_fid), fmt(dit_hair_fid)])
        w.writerow(["Boundary FID ↓", fmt(gan_bnd_fid), fmt(dit_bnd_fid)])
    print(f"\n저장: {out_dir}/fid_summary.csv")


if __name__ == "__main__":
    main()
