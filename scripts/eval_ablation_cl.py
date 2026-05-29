"""
Ablation 3: Curriculum Learning evaluation.

Input: raw hair patches saved by infer_ablation_cl.py
  custom_results/ablation_cl/{tag}/{split}/{stem}.png

GT hair patch: dataset/{split}/img/test/{stem}.png  (matte-masked internally)

평가 그룹:
  unbraid  : C1(Unbraid Only) vs C3(Joint) vs C4(Reverse) vs C5(Ours)
  braid    : C2(Braid Only)   vs C3(Joint) vs C4(Reverse) vs C5(Ours)
  combined : C3(Joint) vs C4(Reverse) vs C5(Ours)  — unbraid+braid 합산

Metrics (hair patch only — boundary/identity 제외):
  [1] Sketch Fidelity : Edge IoU, Chamfer Distance, Sketch LPIPS
  [2] Generation      : Hair FID, LPIPS(GT), SSIM(GT), PSNR(GT)

Outputs:
  eval_results/ablation_cl_{split}_summary.csv
  eval_results/ablation_cl_{split}_per_image.csv

Usage:
  python scripts/eval_ablation_cl.py --split unbraid
  python scripts/eval_ablation_cl.py --split braid
  python scripts/eval_ablation_cl.py --split combined
"""

import argparse
import csv
import math
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 평가 그룹별 variant 정의
VARIANTS_UNBRAID = [
    ("c1_unbraid_only", "Unbraid Only"),
    ("c3_joint",        "Joint"),
    ("c4_reverse",      "Reverse Curriculum"),
    ("c5_ours",         "Ours (Forward)"),
]

VARIANTS_BRAID = [
    ("c2_braid_only", "Braid Only"),
    ("c3_joint",      "Joint"),
    ("c4_reverse",    "Reverse Curriculum"),
    ("c5_ours",       "Ours (Forward)"),
]

VARIANTS_COMBINED = [
    ("c3_joint",   "Joint"),
    ("c4_reverse", "Reverse Curriculum"),
    ("c5_ours",    "Ours (Forward)"),
]

METRIC_KEYS = [
    "edge_iou", "chamfer", "sketch_lpips",
    "psnr", "ssim", "lpips",
    "hair_fid",
]


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_fn = None

def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _lpips_fn


def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    t = t.unsqueeze(0).to(DEVICE)
    if t.shape[-1] < 64 or t.shape[-2] < 64:
        t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
    return t


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    fn = get_lpips()
    with torch.no_grad():
        return float(fn(_to_lpips_tensor(a), _to_lpips_tensor(b)).item())


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * math.log10(255.0 ** 2 / err) if err > 0 else float("inf")


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k   = cv2.getGaussianKernel(11, 1.5)
    k2d = k @ k.T
    f   = lambda x: cv2.filter2D(x, -1, k2d)
    mu_a, mu_b = f(a), f(b)
    sig_a2 = f(a * a) - mu_a ** 2
    sig_b2 = f(b * b) - mu_b ** 2
    sig_ab = f(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
    return float((num / den).mean())


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([_ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------------------
# Masking helpers — GT * matte 및 pred * matte 모두 적용
# ---------------------------------------------------------------------------

def _bbox_crop(img: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy(), mask.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def _hair_crops(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray):
    hair = matte > 127
    pc, mc = _bbox_crop(pred, hair)
    gc, _  = _bbox_crop(gt,   hair)
    pc[~mc] = 0
    gc[~mc] = 0
    return pred[hair], gt[hair], pc, gc


def _fid_crop(img: np.ndarray, mask: np.ndarray, min_px: int = 64):
    if mask.sum() < min_px:
        return None
    crop, m = _bbox_crop(img, mask)
    crop[~m] = 0
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (128, 128))


# ---------------------------------------------------------------------------
# [1] Sketch Fidelity
# ---------------------------------------------------------------------------

def _canny(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150) > 0


def _edge_iou(pred_e: np.ndarray, sk_e: np.ndarray, matte: np.ndarray) -> float:
    hair  = matte > 127
    inter = (pred_e & hair & sk_e).sum()
    union = ((pred_e | sk_e) & hair).sum()
    return float(inter / union) if union > 0 else 0.0


def _chamfer(pred_e: np.ndarray, sk_e: np.ndarray, matte: np.ndarray) -> float:
    hair  = matte > 127
    pts_a = np.argwhere(pred_e & hair).astype(np.float32)
    pts_b = np.argwhere(sk_e   & hair).astype(np.float32)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")
    d_ab, _ = cKDTree(pts_b).query(pts_a, k=1)
    d_ba, _ = cKDTree(pts_a).query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2)


def _sketch_lpips(pred: np.ndarray, sketch: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 127
    ep   = np.stack([_canny(pred).astype(np.uint8) * 255] * 3, axis=-1)
    pc, mc = _bbox_crop(ep,     hair)
    sc, _  = _bbox_crop(sketch, hair)
    pc[~mc] = 0
    sc[~mc] = 0
    if pc.shape[0] < 8 or pc.shape[1] < 8:
        return float("nan")
    return compute_lpips(pc, sc)


# ---------------------------------------------------------------------------
# [2] Generation Quality
# ---------------------------------------------------------------------------

def _gen_metrics(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> dict:
    pv, gv, pc, gc = _hair_crops(pred, gt, matte)
    lpips_val = compute_lpips(pc, gc) if (pc.shape[0] >= 8 and pc.shape[1] >= 8) else float("nan")
    return {"psnr": _psnr(pv, gv), "ssim": _ssim_rgb(pc, gc), "lpips": lpips_val}


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _compute_fid(real_imgs: list, fake_imgs: list) -> float:
    try:
        from pytorch_fid import fid_score
    except ImportError:
        return float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = Path(tmp) / "real"
        fake_dir = Path(tmp) / "fake"
        real_dir.mkdir(); fake_dir.mkdir()
        for i, img in enumerate(real_imgs):
            if img is not None:
                Image.fromarray(img).save(real_dir / f"{i:05d}.png")
        for i, img in enumerate(fake_imgs):
            if img is not None:
                Image.fromarray(img).save(fake_dir / f"{i:05d}.png")
        if len(list(real_dir.glob("*.png"))) < 2 or len(list(fake_dir.glob("*.png"))) < 2:
            return float("nan")
        return float(fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=32, device=str(DEVICE), dims=2048, num_workers=0,
        ))


# ---------------------------------------------------------------------------
# Core per-split evaluation (재사용 가능한 함수)
# ---------------------------------------------------------------------------

def _collect_split(split: str, variants: list) -> tuple:
    """단일 split에 대해 per-image 지표 수집 및 FID 버퍼 반환."""
    gt_img_dir   = ROOT / f"dataset/{split}/img/test"
    gt_matte_dir = ROOT / f"dataset/{split}/matte/test"
    sketch_dir   = ROOT / f"dataset/{split}/sketch/test"

    stems = sorted(p.stem for p in gt_img_dir.glob("*.png"))
    print(f"\n  [{split}] {len(stems)}개")
    if len(stems) < 500:
        print(f"  [WARNING] FID는 500장 이상 권장")

    tags    = [tag for tag, _ in variants]
    fid_buf = {tag: {"real": [], "fake": []} for tag in tags}
    rows    = []

    for stem in tqdm(stems, desc=split):
        gt_img = np.array(Image.open(gt_img_dir   / f"{stem}.png").convert("RGB"))
        matte  = np.array(Image.open(gt_matte_dir  / f"{stem}.png").convert("L"))
        sketch = np.array(Image.open(sketch_dir    / f"{stem}.png").convert("RGB"))

        hair_mask = matte > 127
        gt_fid    = _fid_crop(gt_img, hair_mask)
        sk_edge   = _canny(sketch)
        row       = {"stem": stem, "_split": split}

        for tag, _ in variants:
            pred_path = ROOT / f"custom_results/ablation_cl/{tag}/{split}/{stem}.png"
            if not pred_path.exists():
                for k in METRIC_KEYS:
                    row[f"{tag}_{k}"] = None
                continue

            pred      = np.array(Image.open(pred_path).convert("RGB"))
            pred_edge = _canny(pred)

            row[f"{tag}_edge_iou"]     = _edge_iou(pred_edge, sk_edge, matte)
            row[f"{tag}_chamfer"]      = _chamfer(pred_edge, sk_edge, matte)
            row[f"{tag}_sketch_lpips"] = _sketch_lpips(pred, sketch, matte)

            gq = _gen_metrics(pred, gt_img, matte)
            row[f"{tag}_psnr"]  = gq["psnr"]
            row[f"{tag}_ssim"]  = gq["ssim"]
            row[f"{tag}_lpips"] = gq["lpips"]

            fid_buf[tag]["real"].append(gt_fid)
            fid_buf[tag]["fake"].append(_fid_crop(pred, hair_mask))

        rows.append(row)

    return rows, fid_buf


def _write_results(split_label: str, rows: list, fid_buf: dict, variants: list) -> None:
    """지표 집계 후 CSV 저장 및 콘솔 출력."""
    tags = [tag for tag, _ in variants]

    # FID
    print(f"\n  FID 계산 중 ({split_label})...")
    fid_scores = {}
    for tag in tags:
        real = [x for x in fid_buf[tag]["real"] if x is not None]
        fake = [x for x in fid_buf[tag]["fake"] if x is not None]
        fid  = _compute_fid(real, fake)
        fid_scores[tag] = fid
        print(f"    {tag}: n={len(fake)}, hair_fid={fid:.2f}")

    non_fid_keys = [k for k in METRIC_KEYS if k != "hair_fid"]

    # per-image CSV
    per_img_path = OUT_DIR / f"ablation_cl_{split_label}_per_image.csv"
    fieldnames   = ["stem"] + [f"{tag}_{k}" for tag, _ in variants for k in non_fid_keys]
    with open(per_img_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # summary CSV
    summary_path = OUT_DIR / f"ablation_cl_{split_label}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [label for _, label in variants])
        for k in non_fid_keys:
            vals = []
            for tag, _ in variants:
                col = [r[f"{tag}_{k}"] for r in rows if r.get(f"{tag}_{k}") is not None]
                vals.append(f"{np.mean(col):.4f}" if col else "—")
            writer.writerow([k] + vals)
        writer.writerow(["hair_fid"] + [f"{fid_scores[tag]:.2f}" for tag, _ in variants])

    print(f"\n  summary   → {summary_path}")
    print(f"  per-image → {per_img_path}")

    col_w  = 20
    header = f"  {'metric':<18}" + "".join(f"{label:>{col_w}}" for _, label in variants)
    print(f"\n{header}")
    print("  " + "-" * len(header))
    for k in non_fid_keys:
        vals = []
        for tag, _ in variants:
            col = [r[f"{tag}_{k}"] for r in rows if r.get(f"{tag}_{k}") is not None]
            vals.append(f"{np.mean(col):.4f}" if col else "—")
        print(f"  {k:<18}" + "".join(f"{v:>{col_w}}" for v in vals))
    print(f"  {'hair_fid':<18}" + "".join(f"{fid_scores[tag]:>{col_w}.2f}" for tag, _ in variants))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="unbraid",
                        choices=["unbraid", "braid", "combined"])
    args = parser.parse_args()

    if args.split == "unbraid":
        rows, fid_buf = _collect_split("unbraid", VARIANTS_UNBRAID)
        _write_results("unbraid", rows, fid_buf, VARIANTS_UNBRAID)

    elif args.split == "braid":
        rows, fid_buf = _collect_split("braid", VARIANTS_BRAID)
        _write_results("braid", rows, fid_buf, VARIANTS_BRAID)

    elif args.split == "combined":
        # unbraid + braid 합산, C3/C4/C5만 평가
        rows_u, fid_u = _collect_split("unbraid", VARIANTS_COMBINED)
        rows_b, fid_b = _collect_split("braid",   VARIANTS_COMBINED)

        rows_all = rows_u + rows_b

        # FID 버퍼 합산
        tags = [tag for tag, _ in VARIANTS_COMBINED]
        fid_combined = {tag: {
            "real": fid_u[tag]["real"] + fid_b[tag]["real"],
            "fake": fid_u[tag]["fake"] + fid_b[tag]["fake"],
        } for tag in tags}

        _write_results("combined", rows_all, fid_combined, VARIANTS_COMBINED)


if __name__ == "__main__":
    main()
