"""
Quantitative evaluation: GAN (SHS) vs DiT v1~v4

Systems:
  gan  : custom_results/gan/shs/{stem}.png
  dit1 : custom_results/dit/test_shs_blend/{stem}_full.png
  dit2 : custom_results/dit/weighted_sum_v2/{stem}_full.png
  dit3 : custom_results/dit/weighted_sum_v3/{stem}_full.png
  dit4 : custom_results/dit/weighted_sum_v4/{stem}_full.png

Metrics:
  [1] Sketch Fidelity   : Edge IoU, Chamfer Distance, Sketch LPIPS
  [2] Generation Quality: Hair FID, LPIPS(GT), SSIM(GT), PSNR(GT)
  [3] Boundary Quality  : Boundary FID, Boundary LPIPS
  [4] Identity          : Face LPIPS, ArcFace Cosine (proxy)

Usage:
  python scripts/eval_all.py

Output:
  eval_results/all_summary.csv
  eval_results/all_per_image.csv
"""

import csv
import math
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

ROOT          = Path(__file__).parent.parent
GT_IMG_DIR    = ROOT / "dataset/braid/img/test"
GT_MATTE_DIR  = ROOT / "dataset/braid/matte/test"
SKETCH_DIR    = ROOT / "dataset/braid/sketch/test"
OUT_DIR       = ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

SYSTEMS = [
    ("gan",  ROOT / "custom_results/gan/shs",                    "{stem}.png"),
    ("dit1", ROOT / "custom_results/dit/binary_mask",            "{stem}_full.png"),
    ("dit2", ROOT / "custom_results/dit/weighted_sum_v2",        "{stem}_full.png"),
    ("dit3", ROOT / "custom_results/dit/weighted_sum_v3",        "{stem}_full.png"),
    ("dit4", ROOT / "custom_results/dit/weighted_sum_v4",        "{stem}_full.png"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------
_lpips_fn = None

def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _lpips_fn


def img_to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    fn = get_lpips()
    with torch.no_grad():
        ta = img_to_lpips_tensor(a)
        tb = img_to_lpips_tensor(b)
        if ta.shape[-1] < 64 or ta.shape[-2] < 64:
            ta = F.interpolate(ta, size=(64, 64), mode="bilinear", align_corners=False)
            tb = F.interpolate(tb, size=(64, 64), mode="bilinear", align_corners=False)
        return float(fn(ta, tb).item())


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    err = mse(a, b)
    return 10 * math.log10(255.0 ** 2 / err) if err > 0 else float("inf")


def ssim_channel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    k2d = k @ k.T
    filt = lambda x: cv2.filter2D(x, -1, k2d)
    mu_a, mu_b = filt(a), filt(b)
    sigma_a2 = filt(a * a) - mu_a ** 2
    sigma_b2 = filt(b * b) - mu_b ** 2
    sigma_ab  = filt(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float((num / den).mean())


def ssim_rgb(a, b):
    return float(np.mean([ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def apply_mask_and_crop(img, mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy(), mask.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def masked_metric_arrays(pred, gt, mask):
    hair = mask > 127
    pred_crop, mask_crop = apply_mask_and_crop(pred, hair)
    gt_crop,   _         = apply_mask_and_crop(gt, hair)
    pred_z = pred_crop.copy(); pred_z[~mask_crop] = 0
    gt_z   = gt_crop.copy();   gt_z[~mask_crop]   = 0
    return pred[hair], gt[hair], pred_z, gt_z


def get_boundary_mask(matte, lo=25, hi=230):
    return (matte >= lo) & (matte <= hi)


def extract_region_crop(img, mask, min_px=64):
    if mask.sum() < min_px:
        return None
    crop, m = apply_mask_and_crop(img, mask)
    crop[~m] = 0
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (128, 128))


# ---------------------------------------------------------------------------
# [1] Sketch Fidelity
# ---------------------------------------------------------------------------

def canny_edges(img, low=50, high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high) > 0


def edge_iou(edges_a, edges_b, mask):
    hair = mask > 127
    a_m = edges_a & hair
    b_m = edges_b & hair
    inter = (a_m & b_m).sum()
    union = (a_m | b_m).sum()
    return float(inter / union) if union > 0 else 0.0


def chamfer_distance(edges_a, edges_b, mask):
    hair = mask > 127
    pts_a = np.argwhere(edges_a & hair).astype(np.float32)
    pts_b = np.argwhere(edges_b & hair).astype(np.float32)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")
    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2)


def sketch_lpips(pred, sketch, mask):
    hair = mask > 127
    edge_pred = canny_edges(pred).astype(np.uint8) * 255
    edge_pred_rgb = np.stack([edge_pred] * 3, axis=-1)
    pred_crop, mask_crop = apply_mask_and_crop(edge_pred_rgb, hair)
    sk_crop, _           = apply_mask_and_crop(sketch, hair)
    pred_crop[~mask_crop] = 0
    sk_crop  [~mask_crop] = 0
    if pred_crop.shape[0] < 8 or pred_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(pred_crop, sk_crop)


# ---------------------------------------------------------------------------
# [2] Generation Quality
# ---------------------------------------------------------------------------

def gen_quality_metrics(pred, gt, matte):
    pv, gv, pc, gc = masked_metric_arrays(pred, gt, matte)
    lpips_val = compute_lpips(pc, gc) if (pc.shape[0] >= 8 and pc.shape[1] >= 8) else float("nan")
    return {"psnr": psnr(pv, gv), "ssim": ssim_rgb(pc, gc), "lpips": lpips_val}


# ---------------------------------------------------------------------------
# [3] Boundary Quality
# ---------------------------------------------------------------------------

def boundary_lpips(pred, gt, matte):
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
# [4] Identity
# ---------------------------------------------------------------------------

def face_lpips(pred, gt, matte):
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


_face_model = None
_face_model_name = "none"


def try_load_face_model():
    global _face_model, _face_model_name
    try:
        from facenet_pytorch import InceptionResnetV1
        _face_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        _face_model_name = "facenet-vggface2"
        print("FaceNet (VGGFace2) loaded.")
        return
    except Exception:
        pass
    try:
        import torchvision.models as tvm
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = torch.nn.Identity()
        _face_model = resnet.eval().to(DEVICE)
        _face_model_name = "resnet50-imagenet"
        print("ResNet50 (ImageNet) loaded as face embedding fallback.")
        return
    except Exception as e:
        print(f"Face embedding unavailable ({e})")


def face_embedding(face_img):
    if _face_model is None or face_img.shape[0] < 32 or face_img.shape[1] < 32:
        return None
    size = 160 if "facenet" in _face_model_name else 224
    img_r = cv2.resize(face_img, (size, size))
    t = torch.from_numpy(img_r).float().permute(2, 0, 1) / 255.0
    if "facenet" in _face_model_name:
        t = (t - 0.5) / 0.5
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _face_model(t).cpu().numpy()[0]
    return emb / (np.linalg.norm(emb) + 1e-8)


def arcface_cosine(pred, gt, matte):
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
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_imgs, fake_imgs):
    try:
        from pytorch_fid import fid_score
    except ImportError:
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
        return float(fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=32, device=str(DEVICE), dims=2048, num_workers=0,
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "edge_iou", "chamfer", "sketch_lpips",
    "psnr", "ssim", "lpips",
    "bnd_lpips", "face_lpips", "arcface_cos",
]


def main():
    try_load_face_model()

    stems = sorted([p.stem for p in GT_IMG_DIR.glob("*.png")])
    print(f"\nGT 이미지: {len(stems)}개")
    if len(stems) < 500:
        print(f"[WARNING] FID는 500장 이상 권장 (현재 {len(stems)}장) — 참고용")

    sys_tags = [tag for tag, _, _ in SYSTEMS]

    # FID 버퍼: 각 시스템별로 독립된 real/fake 쌍
    fid_hair  = {tag: {"real": [], "fake": []} for tag in sys_tags}
    fid_bnd   = {tag: {"real": [], "fake": []} for tag in sys_tags}

    rows = []

    for stem in tqdm(stems, desc="Per-image metrics"):
        gt_img  = np.array(Image.open(GT_IMG_DIR  / f"{stem}.png").convert("RGB"))
        matte   = np.array(Image.open(GT_MATTE_DIR / f"{stem}.png").convert("L"))
        sketch  = np.array(Image.open(SKETCH_DIR   / f"{stem}.png").convert("RGB"))

        hair_mask = matte > 127
        bnd_mask  = get_boundary_mask(matte)

        gt_hair_crop = extract_region_crop(gt_img, hair_mask)
        gt_bnd_crop  = extract_region_crop(gt_img, bnd_mask, min_px=16)

        sk_edge = canny_edges(sketch)

        row = {"stem": stem}

        for tag, dirpath, fname_tmpl in SYSTEMS:
            path = dirpath / fname_tmpl.format(stem=stem)
            if not path.exists():
                for k in METRIC_KEYS:
                    row[f"{tag}_{k}"] = None
                continue

            pred = np.array(Image.open(path).convert("RGB"))
            pred_edge = canny_edges(pred)

            # [1] Sketch Fidelity
            row[f"{tag}_edge_iou"]     = edge_iou(pred_edge, sk_edge, matte)
            row[f"{tag}_chamfer"]      = chamfer_distance(pred_edge, sk_edge, matte)
            row[f"{tag}_sketch_lpips"] = sketch_lpips(pred, sketch, matte)

            # [2] Generation Quality
            gq = gen_quality_metrics(pred, gt_img, matte)
            row[f"{tag}_psnr"]  = gq["psnr"]
            row[f"{tag}_ssim"]  = gq["ssim"]
            row[f"{tag}_lpips"] = gq["lpips"]

            # [3] Boundary Quality
            row[f"{tag}_bnd_lpips"] = boundary_lpips(pred, gt_img, matte)

            # [4] Identity
            row[f"{tag}_face_lpips"]  = face_lpips(pred, gt_img, matte)
            row[f"{tag}_arcface_cos"] = arcface_cosine(pred, gt_img, matte)

            # FID 버퍼 (해당 시스템에 이미지가 있을 때만)
            fid_hair[tag]["real"].append(gt_hair_crop)
            fid_hair[tag]["fake"].append(extract_region_crop(pred, hair_mask))
            fid_bnd[tag]["real"].append(gt_bnd_crop)
            fid_bnd[tag]["fake"].append(extract_region_crop(pred, bnd_mask, min_px=16))

        rows.append(row)

    # --- FID ---
    print("\nFID 계산 중...")
    fid_results = {}
    for tag in sys_tags:
        real_h = [x for x in fid_hair[tag]["real"] if x is not None]
        fake_h = [x for x in fid_hair[tag]["fake"] if x is not None]
        real_b = [x for x in fid_bnd[tag]["real"]  if x is not None]
        fake_b = [x for x in fid_bnd[tag]["fake"]  if x is not None]
        fid_results[f"{tag}_hair_fid"] = compute_fid(real_h, fake_h)
        fid_results[f"{tag}_bnd_fid"]  = compute_fid(real_b, fake_b)
        n = len(fake_h)
        print(f"  {tag}: n={n}, hair_fid={fid_results[f'{tag}_hair_fid']:.2f}, bnd_fid={fid_results[f'{tag}_bnd_fid']:.2f}")

    # --- Per-image CSV ---
    per_path = OUT_DIR / "all_per_image.csv"
    if rows:
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

    def n_avail(tag):
        return sum(1 for r in rows if r.get(f"{tag}_psnr") is not None)

    col_labels = ["GAN(SHS)", "DiT v1", "DiT v2", "DiT v3", "DiT v4"]
    col_width = 12

    metric_groups = [
        ("Sketch Fidelity", [
            ("Edge IoU ↑",       "edge_iou",    None),
            ("Chamfer Dist ↓",   "chamfer",     None),
            ("Sketch LPIPS ↓",   "sketch_lpips",None),
        ]),
        ("Generation Quality", [
            ("Hair FID ↓",       None,          "hair_fid"),
            ("LPIPS (GT) ↓",     "lpips",       None),
            ("SSIM (GT) ↑",      "ssim",        None),
            ("PSNR (GT) ↑",      "psnr",        None),
        ]),
        ("Boundary Quality", [
            ("Boundary FID ↓",   None,          "bnd_fid"),
            ("Boundary LPIPS ↓", "bnd_lpips",   None),
        ]),
        ("Identity", [
            ("Face LPIPS ↓",     "face_lpips",  None),
            ("ArcFace Cos ↑",    "arcface_cos", None),
        ]),
    ]

    header_line = f"{'Metric':<22}" + "".join(f"  {lbl:>{col_width}}" for lbl in col_labels)
    sep = "=" * (22 + (col_width + 2) * len(col_labels))

    print(f"\n{sep}")
    print(header_line)
    print(sep)

    summary_rows = []
    for group_name, metrics in metric_groups:
        print(f"\n--- {group_name} ---")
        for label, per_key, fid_key in metrics:
            vals = []
            for tag in sys_tags:
                if fid_key:
                    v = fid_results.get(f"{tag}_{fid_key}")
                else:
                    v = safe_mean([r.get(f"{tag}_{per_key}") for r in rows])
                vals.append(v)
            row_str = f"  {label:<20}" + "".join(f"  {fmt(v):>{col_width}}" for v in vals)
            print(row_str)
            summary_rows.append([label] + [fmt(v) for v in vals])

    print(f"\n{sep}")
    ns = [f"{tag}(n={n_avail(tag)})" for tag in sys_tags]
    print("Sample sizes: " + ", ".join(ns))

    sum_path = OUT_DIR / "all_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + col_labels)
        w.writerows(summary_rows)

    print(f"\n저장 완료:\n  {per_path}\n  {sum_path}")


if __name__ == "__main__":
    main()
