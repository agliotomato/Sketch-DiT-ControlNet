"""
Full quantitative evaluation: GAN vs DiT (test_batch2/full.png)

[1] Sketch Fidelity   : Edge IoU, Chamfer Distance, Sketch LPIPS
[2] Generation Quality: FID, LPIPS(vs GT), SSIM(vs GT), PSNR(vs GT)
[3] Boundary Quality  : Boundary FID, Boundary LPIPS
[4] Identity          : Face LPIPS, FaceNet cosine (ArcFace proxy)

Usage:
  python scripts/eval_full.py

Output:
  eval_results/full_summary.csv
  eval_results/full_per_image.csv
"""

import csv
import math
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GT_IMG_DIR    = Path("dataset/braid/img/test")
GT_MATTE_DIR  = Path("dataset/braid/matte/test")
SKETCH_DIR    = Path("dataset/braid/sketch/test")
GAN_DIR       = Path("custom_results/gan/shs")
TB1_DIR       = Path("custom_results/test_batch")
TB2_DIR       = Path("custom_results/test_batch2")
OUT_DIR       = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# LPIPS model (alex network, cached after first call)
# ---------------------------------------------------------------------------
_lpips_fn = None

def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _lpips_fn


def img_to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 RGB HWC → [-1,1] BCHW tensor."""
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS on full RGB images (uint8 HWC)."""
    fn = get_lpips()
    with torch.no_grad():
        ta = img_to_lpips_tensor(a)
        tb = img_to_lpips_tensor(b)
        # Resize to 64x64 minimum for LPIPS (alex requires ≥64)
        if ta.shape[-1] < 64 or ta.shape[-2] < 64:
            ta = F.interpolate(ta, size=(64, 64), mode="bilinear", align_corners=False)
            tb = F.interpolate(tb, size=(64, 64), mode="bilinear", align_corners=False)
        return float(fn(ta, tb).item())


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = mse(a, b)
    return 10 * math.log10(255.0 ** 2 / err) if err > 0 else float("inf")


def ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    k2d = k @ k.T
    filt = lambda x: cv2.filter2D(x, -1, k2d)
    mu_a, mu_b = filt(a), filt(b)
    sigma_a2 = filt(a * a) - mu_a ** 2
    sigma_b2 = filt(b * b) - mu_b ** 2
    sigma_ab = filt(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float((num / den).mean())


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def get_hair_mask(matte: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Boolean 2D mask for hair region."""
    return matte > thresh


def apply_mask_and_crop(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to bounding box of mask region."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy(), mask.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def masked_metric_arrays(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Return (pred_pixels, gt_pixels, pred_crop, gt_crop) for hair region."""
    hair = mask > 127
    pred_crop, mask_crop = apply_mask_and_crop(pred, hair)
    gt_crop,   _         = apply_mask_and_crop(gt, hair)
    # zero out non-hair in crop for spatial metrics (SSIM, LPIPS)
    pred_z = pred_crop.copy(); pred_z[~mask_crop] = 0
    gt_z   = gt_crop.copy();   gt_z[~mask_crop]   = 0
    return pred[hair], gt[hair], pred_z, gt_z


# ---------------------------------------------------------------------------
# [1] Sketch Fidelity
# ---------------------------------------------------------------------------

def canny_edges(img: np.ndarray, low=50, high=150) -> np.ndarray:
    """Canny edge map from RGB uint8 image. Returns bool HW."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high) > 0


def edge_iou(edges_a: np.ndarray, edges_b: np.ndarray, mask: np.ndarray) -> float:
    """IoU of two edge maps within hair mask."""
    hair = mask > 127
    a_m = edges_a & hair
    b_m = edges_b & hair
    inter = (a_m & b_m).sum()
    union = (a_m | b_m).sum()
    return float(inter / union) if union > 0 else 0.0


def chamfer_distance(edges_a: np.ndarray, edges_b: np.ndarray, mask: np.ndarray) -> float:
    """Bidirectional Chamfer distance between two edge sets within hair mask."""
    hair = mask > 127
    pts_a = np.argwhere(edges_a & hair).astype(np.float32)
    pts_b = np.argwhere(edges_b & hair).astype(np.float32)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2)


def sketch_lpips(pred: np.ndarray, sketch: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS between Canny(pred) and sketch, in hair region bounding box."""
    hair = mask > 127
    # edge map of prediction (3-channel for LPIPS)
    edge_pred = canny_edges(pred).astype(np.uint8) * 255
    edge_pred_rgb = np.stack([edge_pred] * 3, axis=-1)

    pred_crop, mask_crop = apply_mask_and_crop(edge_pred_rgb, hair)
    sk_crop, _           = apply_mask_and_crop(sketch, hair)

    # zero non-hair
    pred_crop[~mask_crop] = 0
    sk_crop  [~mask_crop] = 0

    if pred_crop.shape[0] < 8 or pred_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(pred_crop, sk_crop)


# ---------------------------------------------------------------------------
# [2] Generation Quality — per-image (FID done separately)
# ---------------------------------------------------------------------------

def gen_quality_metrics(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> dict:
    pv, gv, pc, gc = masked_metric_arrays(pred, gt, matte)
    lpips_val = compute_lpips(pc, gc) if (pc.shape[0] >= 8 and pc.shape[1] >= 8) else float("nan")
    return {
        "psnr": psnr(pv, gv),
        "ssim": ssim_rgb(pc, gc),
        "lpips": lpips_val,
    }


# ---------------------------------------------------------------------------
# [3] Boundary Quality — per-image (boundary FID done separately)
# ---------------------------------------------------------------------------

def get_boundary_mask(matte: np.ndarray, lo=25, hi=230) -> np.ndarray:
    """Boundary strip: matte values in [lo, hi] (soft region ≈ 0.1~0.9)."""
    return (matte >= lo) & (matte <= hi)


def boundary_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    bnd = get_boundary_mask(matte)
    if bnd.sum() < 64:
        return float("nan")
    pred_crop, bnd_crop = apply_mask_and_crop(pred, bnd)
    gt_crop, _          = apply_mask_and_crop(gt, bnd)
    pred_crop[~bnd_crop] = 0
    gt_crop  [~bnd_crop] = 0
    if pred_crop.shape[0] < 8 or pred_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(pred_crop, gt_crop)


# ---------------------------------------------------------------------------
# [4] Identity — per-image (ArcFace done separately if available)
# ---------------------------------------------------------------------------

def face_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    """LPIPS on face region (1 - hair_mask)."""
    face_mask = matte < 127
    if face_mask.sum() < 64:
        return float("nan")
    pred_crop, fc = apply_mask_and_crop(pred, face_mask)
    gt_crop,   _  = apply_mask_and_crop(gt, face_mask)
    pred_crop[~fc] = 0
    gt_crop  [~fc] = 0
    if pred_crop.shape[0] < 8 or pred_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(pred_crop, gt_crop)


# ---------------------------------------------------------------------------
# FaceNet-based identity (proxy for ArcFace cosine)
# ---------------------------------------------------------------------------
_face_model = None
_face_model_name = "none"

def try_load_face_model():
    global _face_model, _face_model_name

    # 1순위: facenet-pytorch (ArcFace 스타일 학습 임베딩)
    try:
        from facenet_pytorch import InceptionResnetV1
        _face_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        _face_model_name = "facenet-vggface2"
        print("FaceNet (VGGFace2) loaded as ArcFace proxy.")
        return
    except Exception:
        pass

    # 2순위: torchvision ResNet50 (일반 임베딩 — identity 보존 proxy)
    try:
        import torchvision.models as tvm
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # 마지막 fc 제거 → 2048-d feature
        resnet.fc = torch.nn.Identity()
        _face_model = resnet.eval().to(DEVICE)
        _face_model_name = "resnet50-imagenet"
        print("ResNet50 (ImageNet) loaded as face embedding fallback.")
        return
    except Exception as e:
        print(f"Face embedding unavailable ({e}); ArcFace cosine will be N/A.")


def face_embedding(face_img: np.ndarray) -> np.ndarray | None:
    """face_img: uint8 RGB HWC. Returns L2-normalized embedding or None."""
    if _face_model is None or face_img.shape[0] < 32 or face_img.shape[1] < 32:
        return None
    size = 160 if "facenet" in _face_model_name else 224
    img_resized = cv2.resize(face_img, (size, size))
    t = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    if "facenet" in _face_model_name:
        t = (t - 0.5) / 0.5
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _face_model(t)
    emb = emb.cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def arcface_cosine(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    """Cosine similarity of face embeddings in face region."""
    if _face_model is None:
        return float("nan")
    face_mask = matte < 127
    pred_face, fc = apply_mask_and_crop(pred, face_mask)
    gt_face,   _  = apply_mask_and_crop(gt, face_mask)
    emb_pred = face_embedding(pred_face)
    emb_gt   = face_embedding(gt_face)
    if emb_pred is None or emb_gt is None:
        return float("nan")
    return float(np.dot(emb_pred, emb_gt))


# ---------------------------------------------------------------------------
# FID computation (batch, uses temp dirs)
# ---------------------------------------------------------------------------

def compute_fid(real_imgs: list[np.ndarray], fake_imgs: list[np.ndarray], label: str) -> float:
    """Save crops to temp dirs, run pytorch-fid."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("pytorch-fid not available; FID skipped.")
        return float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        real_dir = Path(tmp) / "real"
        fake_dir = Path(tmp) / "fake"
        real_dir.mkdir(); fake_dir.mkdir()

        for i, img in enumerate(real_imgs):
            if img is not None and img.size > 0:
                Image.fromarray(img).save(real_dir / f"{i:05d}.png")
        for i, img in enumerate(fake_imgs):
            if img is not None and img.size > 0:
                Image.fromarray(img).save(fake_dir / f"{i:05d}.png")

        n_real = len(list(real_dir.glob("*.png")))
        n_fake = len(list(fake_dir.glob("*.png")))
        if n_real < 2 or n_fake < 2:
            return float("nan")

        fid_val = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=32,
            device=str(DEVICE),
            dims=2048,
            num_workers=0,
        )
    return float(fid_val)


def extract_region_crop(img: np.ndarray, mask: np.ndarray, min_px: int = 64) -> np.ndarray | None:
    """Extract region crop, resize to 128x128 for FID consistency."""
    if mask.sum() < min_px:
        return None
    crop, m = apply_mask_and_crop(img, mask)
    crop[~m] = 0
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (128, 128))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try_load_face_model()

    stems = sorted([p.stem for p in GT_IMG_DIR.glob("*.png")])
    print(f"\n평가 이미지: {len(stems)}개")
    if len(stems) < 500:
        print(f"[WARNING] FID requires 500+ images (current: {len(stems)}) -- reference only")

    rows = []

    # FID를 위한 crop 버퍼
    fid_bufs = {
        "hair_real": [], "hair_gan": [], "hair_dit": [],
        "bnd_real":  [], "bnd_gan":  [], "bnd_dit":  [],
    }

    for stem in tqdm(stems, desc="Per-image metrics"):
        gt_img   = np.array(Image.open(GT_IMG_DIR   / f"{stem}.png").convert("RGB"))
        matte    = np.array(Image.open(GT_MATTE_DIR  / f"{stem}.png").convert("L"))
        sketch   = np.array(Image.open(SKETCH_DIR    / f"{stem}.png").convert("RGB"))
        hair_mask = matte > 127
        bnd_mask  = get_boundary_mask(matte)

        gan_path = GAN_DIR / f"{stem}.png"
        dit_path = DIT_DIR / f"{stem}_full.png"

        gan_img = np.array(Image.open(gan_path).convert("RGB")) if gan_path.exists() else None
        dit_img = np.array(Image.open(dit_path).convert("RGB")) if dit_path.exists() else None

        # FID buffers
        fid_bufs["hair_real"].append(extract_region_crop(gt_img, hair_mask))
        fid_bufs["bnd_real"].append(extract_region_crop(gt_img, bnd_mask, min_px=16))

        row = {"stem": stem}

        for tag, pred in [("gan", gan_img), ("dit", dit_img)]:
            if pred is None:
                for k in ["edge_iou", "chamfer", "sketch_lpips",
                          "psnr", "ssim", "lpips",
                          "bnd_lpips", "face_lpips", "arcface_cos"]:
                    row[f"{tag}_{k}"] = None
                fid_bufs[f"hair_{tag}"].append(None)
                fid_bufs[f"bnd_{tag}"].append(None)
                continue

            # --- [1] Sketch Fidelity ---
            sk_edge   = canny_edges(sketch)
            pred_edge = canny_edges(pred)
            row[f"{tag}_edge_iou"]     = edge_iou(pred_edge, sk_edge, matte)
            row[f"{tag}_chamfer"]      = chamfer_distance(pred_edge, sk_edge, matte)
            row[f"{tag}_sketch_lpips"] = sketch_lpips(pred, sketch, matte)

            # --- [2] Generation Quality ---
            gq = gen_quality_metrics(pred, gt_img, matte)
            row[f"{tag}_psnr"]  = gq["psnr"]
            row[f"{tag}_ssim"]  = gq["ssim"]
            row[f"{tag}_lpips"] = gq["lpips"]

            # --- [3] Boundary Quality ---
            row[f"{tag}_bnd_lpips"] = boundary_lpips(pred, gt_img, matte)

            # --- [4] Identity ---
            row[f"{tag}_face_lpips"]   = face_lpips(pred, gt_img, matte)
            row[f"{tag}_arcface_cos"]  = arcface_cosine(pred, gt_img, matte)

            # FID crops
            fid_bufs[f"hair_{tag}"].append(extract_region_crop(pred, hair_mask))
            fid_bufs[f"bnd_{tag}"].append(extract_region_crop(pred, bnd_mask, min_px=16))

        rows.append(row)

    # --- FID (batch) ---
    print("\nFID 계산 중...")
    real_hair = [x for x in fid_bufs["hair_real"] if x is not None]
    real_bnd  = [x for x in fid_bufs["bnd_real"]  if x is not None]

    fid_results = {}
    for tag in ["gan", "dit"]:
        fake_hair = [x for x in fid_bufs[f"hair_{tag}"] if x is not None]
        fake_bnd  = [x for x in fid_bufs[f"bnd_{tag}"]  if x is not None]
        fid_results[f"{tag}_hair_fid"] = compute_fid(real_hair, fake_hair, f"{tag} hair")
        fid_results[f"{tag}_bnd_fid"]  = compute_fid(real_bnd,  fake_bnd,  f"{tag} boundary")

    # --- Save per-image CSV ---
    per_path = OUT_DIR / "full_per_image.csv"
    fields = list(rows[0].keys())
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # --- Summary ---
    def safe_mean(vals):
        vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(vs) / len(vs) if vs else None

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    metric_groups = [
        ("--- Sketch Fidelity ---", [
            ("Edge IoU ↑",       "edge_iou"),
            ("Chamfer Dist ↓",   "chamfer"),
            ("Sketch LPIPS ↓",   "sketch_lpips"),
        ]),
        ("--- Generation Quality ---", [
            ("Hair FID ↓",       None),  # from fid_results
            ("LPIPS (GT) ↓",     "lpips"),
            ("SSIM (GT) ↑",      "ssim"),
            ("PSNR (GT) ↑",      "psnr"),
        ]),
        ("--- Boundary Quality ---", [
            ("Boundary FID ↓",   None),  # from fid_results
            ("Boundary LPIPS ↓", "bnd_lpips"),
        ]),
        ("--- Identity ---", [
            ("Face LPIPS ↓",     "face_lpips"),
            ("ArcFace Cos ↑",    "arcface_cos"),
        ]),
    ]

    summary_rows = []
    print("\n" + "=" * 62)
    print(f"{'Metric':<22}  {'GAN':>12}  {'DiT (full)':>12}")
    print("=" * 62)

    for group_name, metrics in metric_groups:
        print(f"\n{group_name}")
        for label, key in metrics:
            if key is None:
                # FID
                if "Hair" in label:
                    gv = fid_results.get("gan_hair_fid")
                    dv = fid_results.get("dit_hair_fid")
                else:
                    gv = fid_results.get("gan_bnd_fid")
                    dv = fid_results.get("dit_bnd_fid")
            else:
                gv = safe_mean([r.get(f"gan_{key}") for r in rows])
                dv = safe_mean([r.get(f"dit_{key}") for r in rows])
            print(f"  {label:<20}  {fmt(gv):>12}  {fmt(dv):>12}")
            summary_rows.append([label, fmt(gv), fmt(dv)])

    print("\n" + "=" * 62)
    print(f"(n={len(stems)}, hair region masked)")

    # Save summary CSV
    sum_path = OUT_DIR / "full_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "GAN", "DiT (full)"])
        w.writerows(summary_rows)

    print(f"\n저장 완료:\n  {per_path}\n  {sum_path}")


if __name__ == "__main__":
    main()
