"""
정량 평가: GAN vs DiT (test_batch2 full.png) — GT 대비

평가 지표 (hair region 내):
  MSE, PSNR, SSIM, MAD (Mean Absolute Difference)

Usage:
  python scripts/eval_quantitative.py

출력:
  eval_results/quantitative_summary.csv
  eval_results/quantitative_per_image.csv
  콘솔 요약 테이블
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GT_IMG_DIR    = Path("dataset/braid/img/test")
GT_MATTE_DIR  = Path("dataset/braid/matte/test")
GAN_DIR       = Path("custom_results/gan/shs")
DIT_DIR       = Path("custom_results/test_batch2")
OUT_DIR       = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return 10 * math.log10(255.0 ** 2 / err)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM (Wang et al., 2004). a, b: uint8 2D."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel2d = kernel @ kernel.T

    def filt(x):
        return cv2.filter2D(x, -1, kernel2d)

    mu_a = filt(a)
    mu_b = filt(b)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b
    sigma_a2 = filt(a ** 2) - mu_a2
    sigma_b2 = filt(b ** 2) - mu_b2
    sigma_ab = filt(a * b) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2))
    return float(ssim_map.mean())


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM across R, G, B channels."""
    return float(np.mean([ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)]))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    pred, gt: uint8 RGB (H, W, 3)
    mask: uint8 L (H, W), 0~255 — hair region
    """
    if mask is not None:
        hair = mask > 127
        if hair.sum() == 0:
            return {"mse": None, "psnr": None, "mad": None, "ssim": None}
        pred_m = pred[hair]
        gt_m = gt[hair]
        # SSIM은 공간 구조가 필요해 → masked 직사각형 crop으로 계산
        ys, xs = np.where(hair)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        pred_crop = pred[y0:y1, x0:x1]
        gt_crop   = gt[y0:y1, x0:x1]
        mask_crop = hair[y0:y1, x0:x1]
        # crop 이미지에서 비-hair 픽셀은 동일 값으로 채워 SSIM 왜곡 최소화
        pred_crop = pred_crop.copy()
        gt_crop   = gt_crop.copy()
        pred_crop[~mask_crop] = 0
        gt_crop[~mask_crop] = 0
    else:
        pred_m = pred.reshape(-1, 3)
        gt_m = gt.reshape(-1, 3)
        pred_crop, gt_crop = pred, gt

    return {
        "mse":  mse(pred_m, gt_m),
        "psnr": psnr(pred_m, gt_m),
        "mad":  mad(pred_m, gt_m),
        "ssim": ssim_rgb(pred_crop, gt_crop),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stems = sorted([p.stem for p in GT_IMG_DIR.glob("*.png")])
    print(f"평가 이미지: {len(stems)}개\n")

    rows = []
    missing_gan = []
    missing_dit = []

    for stem in stems:
        gt_img_path   = GT_IMG_DIR  / f"{stem}.png"
        gt_mask_path  = GT_MATTE_DIR / f"{stem}.png"
        gan_path      = GAN_DIR     / f"{stem}.png"
        dit_path      = DIT_DIR     / f"{stem}_full.png"

        gt_img  = cv2.cvtColor(cv2.imread(str(gt_img_path)),  cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)

        # GAN
        if gan_path.exists():
            gan_img = cv2.cvtColor(cv2.imread(str(gan_path)), cv2.COLOR_BGR2RGB)
            gan_m = compute_metrics(gan_img, gt_img, gt_mask)
        else:
            missing_gan.append(stem)
            gan_m = {"mse": None, "psnr": None, "mad": None, "ssim": None}

        # DiT full
        if dit_path.exists():
            dit_img = cv2.cvtColor(cv2.imread(str(dit_path)), cv2.COLOR_BGR2RGB)
            dit_m = compute_metrics(dit_img, gt_img, gt_mask)
        else:
            missing_dit.append(stem)
            dit_m = {"mse": None, "psnr": None, "mad": None, "ssim": None}

        rows.append({
            "stem":      stem,
            "gan_mse":   gan_m["mse"],
            "gan_psnr":  gan_m["psnr"],
            "gan_mad":   gan_m["mad"],
            "gan_ssim":  gan_m["ssim"],
            "dit_mse":   dit_m["mse"],
            "dit_psnr":  dit_m["psnr"],
            "dit_mad":   dit_m["mad"],
            "dit_ssim":  dit_m["ssim"],
        })

    # 개별 결과 CSV
    per_path = OUT_DIR / "quantitative_per_image.csv"
    with open(per_path, "w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 요약
    def safe_mean(vals):
        vs = [v for v in vals if v is not None]
        return sum(vs) / len(vs) if vs else None

    metrics = ["mse", "psnr", "mad", "ssim"]
    summary = {}
    for model in ["gan", "dit"]:
        for m in metrics:
            key = f"{model}_{m}"
            vals = [r[key] for r in rows]
            summary[key] = safe_mean(vals)

    sum_path = OUT_DIR / "quantitative_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "GAN", "DiT (full)"])
        for m in metrics:
            gv = summary[f"gan_{m}"]
            dv = summary[f"dit_{m}"]
            w.writerow([
                m.upper(),
                f"{gv:.4f}" if gv is not None else "N/A",
                f"{dv:.4f}" if dv is not None else "N/A",
            ])

    # 콘솔 출력
    print("=" * 52)
    print(f"{'Metric':<10}  {'GAN':>12}  {'DiT (full)':>12}")
    print("-" * 52)
    labels = {"mse": "MSE ↓", "psnr": "PSNR ↑", "mad": "MAD ↓", "ssim": "SSIM ↑"}
    for m in metrics:
        gv = summary[f"gan_{m}"]
        dv = summary[f"dit_{m}"]
        gstr = f"{gv:.4f}" if gv is not None else "N/A"
        dstr = f"{dv:.4f}" if dv is not None else "N/A"
        print(f"{labels[m]:<10}  {gstr:>12}  {dstr:>12}")
    print("=" * 52)
    print(f"\n(hair region 마스크 내 평가, n={len([r for r in rows if r['gan_mse'] is not None])} / {len(stems)})")

    if missing_gan:
        print(f"\n[WARNING] GAN 결과 없음 ({len(missing_gan)}개): {missing_gan[:5]}{'...' if len(missing_gan)>5 else ''}")
    if missing_dit:
        print(f"[WARNING] DiT full 없음 ({len(missing_dit)}개): {missing_dit[:5]}{'...' if len(missing_dit)>5 else ''}")

    print(f"\n저장: {per_path}, {sum_path}")


if __name__ == "__main__":
    main()
