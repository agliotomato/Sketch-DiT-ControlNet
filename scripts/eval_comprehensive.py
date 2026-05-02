"""
Comprehensive quantitative evaluation: SHS (GAN) vs Ours (DiT)

Braid split   : SHS vs DiT v2 vs DiT v4  (3-way, in-domain)
Unbraid split : SHS vs DiT (braid ckp)   (2-way, cross-domain zero-shot)
Combined      : SHS vs DiT v4 + braid-ckp (2-way, braid=v4 / unbraid=braid-ckp)

Metrics (same as eval_all.py):
  [1] Sketch Fidelity   : Edge IoU, Chamfer Distance, Sketch LPIPS
  [2] Generation Quality: Hair FID, LPIPS(GT), SSIM(GT), PSNR(GT)
  [3] Boundary Quality  : Boundary FID, Boundary LPIPS
  [4] Identity          : Face LPIPS, ArcFace Cosine

Usage:
  python scripts/eval_comprehensive.py

Output:
  eval_results/braid_summary.csv         (3-col: SHS / DiT v2 / DiT v4)
  eval_results/braid_per_image.csv
  eval_results/unbraid_summary.csv       (2-col: SHS / DiT)
  eval_results/unbraid_per_image.csv
  eval_results/combined_summary.csv      (2-col: SHS / DiT v4)
  eval_results/combined_per_image.csv
"""

import csv
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
import eval_all as M  # noqa: E402

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

# Each split defines: gt dirs + a list of (tag, dir, filename-template)
SPLITS = {
    "braid": {
        "gt_img":   ROOT / "dataset/braid/img/test",
        "gt_matte": ROOT / "dataset/braid/matte/test",
        "sketch":   ROOT / "dataset/braid/sketch/test",
        "systems": [
            ("shs",    ROOT / "custom_results/gan/shs",             "{stem}.png"),
            ("dit_v2", ROOT / "custom_results/dit/weighted_sum_v2", "{stem}_full.png"),
            ("dit_v4", ROOT / "custom_results/dit/weighted_sum_v4", "{stem}_full.png"),
        ],
        "col_labels": ["SHS (GAN)", "DiT v2", "DiT v4"],
    },
    "unbraid": {
        "gt_img":   ROOT / "dataset/unbraid/img/test",
        "gt_matte": ROOT / "dataset/unbraid/matte/test",
        "sketch":   ROOT / "dataset/unbraid/sketch/test",
        "systems": [
            ("shs",  ROOT / "custom_results/gan/generated_unbraid",       "{stem}.png"),
            ("ours", ROOT / "custom_results/dit/inferunbraidbybraidckp",  "{stem}_full.png"),
        ],
        "col_labels": ["SHS (GAN)", "Ours (DiT)"],
    },
}

# Combined uses: braid=dit_v4, unbraid=ours  (consistent single "best Ours" column)
COMBINED_TAGS   = ("shs", "ours")
COMBINED_LABELS = ["SHS (GAN)", "Ours (DiT)"]

METRIC_KEYS = [
    "edge_iou", "chamfer", "sketch_lpips",
    "psnr", "ssim", "lpips",
    "bnd_lpips", "face_lpips", "arcface_cos",
]

METRIC_GROUPS = [
    ("Sketch Fidelity", [
        ("Edge IoU ↑",       "edge_iou",     None,       True),
        ("Chamfer Dist ↓",   "chamfer",      None,       False),
        ("Sketch LPIPS ↓",   "sketch_lpips", None,       False),
    ]),
    ("Generation Quality", [
        ("Hair FID ↓",       None,           "hair_fid", False),
        ("LPIPS (GT) ↓",     "lpips",        None,       False),
        ("SSIM (GT) ↑",      "ssim",         None,       True),
        ("PSNR (GT) ↑",      "psnr",         None,       True),
    ]),
    ("Boundary Quality", [
        ("Boundary FID ↓",   None,           "bnd_fid",  False),
        ("Boundary LPIPS ↓", "bnd_lpips",    None,       False),
    ]),
    ("Identity", [
        ("Face LPIPS ↓",     "face_lpips",   None,       False),
        ("ArcFace Cos ↑",    "arcface_cos",  None,       True),
    ]),
]


# ---------------------------------------------------------------------------
# Stem discovery
# ---------------------------------------------------------------------------

def discover_stems(cfg: dict) -> list:
    gt_stems = {p.stem for p in cfg["gt_img"].glob("*.png")}

    sys_stems = []
    for tag, dirpath, tmpl in cfg["systems"]:
        if "_full" in tmpl:
            stems = {p.stem[:-5] for p in dirpath.glob("*_full.png")}
        else:
            stems = {p.stem for p in dirpath.glob("*.png")}
        sys_stems.append(stems)

    common = gt_stems
    for s in sys_stems:
        common = common & s

    missing = gt_stems - common
    if missing:
        print(f"  [INFO] 일부 시스템에 없어 제외된 stems: {len(missing)}개")

    valid = []
    for stem in sorted(common):
        if not (cfg["gt_matte"] / f"{stem}.png").exists():
            print(f"  [WARN] matte 없음: {stem}")
            continue
        if not (cfg["sketch"] / f"{stem}.png").exists():
            print(f"  [WARN] sketch 없음: {stem}")
            continue
        valid.append(stem)
    return valid


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Path, target_hw=None) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    if target_hw is not None and img.shape[:2] != tuple(target_hw):
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------
# Per-split evaluation (variable number of systems)
# ---------------------------------------------------------------------------

def evaluate_split(split_name: str, cfg: dict, stems: list):
    """
    Returns:
        rows     : list of per-image dicts  {stem, split, {tag}_metric, ...}
        fid_bufs : {tag: {"hair_real", "hair_fake", "bnd_real", "bnd_fake"}}
    """
    tags = [tag for tag, _, _ in cfg["systems"]]
    fid_bufs = {tag: {"hair_real": [], "hair_fake": [], "bnd_real": [], "bnd_fake": []}
                for tag in tags}
    rows = []

    for stem in tqdm(stems, desc=f"  {split_name}"):
        gt_img = load_image(cfg["gt_img"]   / f"{stem}.png")
        matte  = np.array(Image.open(cfg["gt_matte"] / f"{stem}.png").convert("L"))
        sketch = load_image(cfg["sketch"]    / f"{stem}.png")
        hw = gt_img.shape[:2]

        hair_mask = matte > 127
        bnd_mask  = M.get_boundary_mask(matte)
        sk_edge   = M.canny_edges(sketch)

        gt_hair_crop = M.extract_region_crop(gt_img, hair_mask)
        gt_bnd_crop  = M.extract_region_crop(gt_img, bnd_mask, min_px=16)

        row = {"stem": stem, "split": split_name}

        for tag, dirpath, tmpl in cfg["systems"]:
            path = dirpath / tmpl.format(stem=stem)
            if not path.exists():
                for k in METRIC_KEYS:
                    row[f"{tag}_{k}"] = None
                continue

            pred      = load_image(path, target_hw=hw)
            pred_edge = M.canny_edges(pred)

            row[f"{tag}_edge_iou"]     = M.edge_iou(pred_edge, sk_edge, matte)
            row[f"{tag}_chamfer"]      = M.chamfer_distance(pred_edge, sk_edge, matte)
            row[f"{tag}_sketch_lpips"] = M.sketch_lpips(pred, sketch, matte)

            gq = M.gen_quality_metrics(pred, gt_img, matte)
            row[f"{tag}_psnr"]  = gq["psnr"]
            row[f"{tag}_ssim"]  = gq["ssim"]
            row[f"{tag}_lpips"] = gq["lpips"]

            row[f"{tag}_bnd_lpips"]   = M.boundary_lpips(pred, gt_img, matte)
            row[f"{tag}_face_lpips"]  = M.face_lpips(pred, gt_img, matte)
            row[f"{tag}_arcface_cos"] = M.arcface_cosine(pred, gt_img, matte)

            fid_bufs[tag]["hair_real"].append(gt_hair_crop)
            fid_bufs[tag]["hair_fake"].append(M.extract_region_crop(pred, hair_mask))
            fid_bufs[tag]["bnd_real"].append(gt_bnd_crop)
            fid_bufs[tag]["bnd_fake"].append(M.extract_region_crop(pred, bnd_mask, min_px=16))

        rows.append(row)

    return rows, fid_bufs


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid_for_split(fid_bufs: dict) -> dict:
    def valid(lst):
        return [x for x in lst if x is not None]

    results = {}
    for tag, buf in fid_bufs.items():
        results[f"{tag}_hair_fid"] = M.compute_fid(valid(buf["hair_real"]), valid(buf["hair_fake"]))
        results[f"{tag}_bnd_fid"]  = M.compute_fid(valid(buf["bnd_real"]),  valid(buf["bnd_fake"]))
    return results


# ---------------------------------------------------------------------------
# Summary aggregation (variable systems)
# ---------------------------------------------------------------------------

def safe_mean(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vs) / len(vs) if vs else None


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def aggregate_summary(rows: list, fid_results: dict, tags: list) -> list:
    summary = []
    for _, metrics in METRIC_GROUPS:
        for label, per_key, fid_key, _ in metrics:
            vals = []
            for tag in tags:
                if fid_key:
                    v = fid_results.get(f"{tag}_{fid_key}")
                else:
                    v = safe_mean([r.get(f"{tag}_{per_key}") for r in rows])
                vals.append(v)
            summary.append([label] + [fmt(v) for v in vals])
    return summary


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_per_image_csv(rows: list, path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(summary_rows: list, path: Path, col_labels: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + col_labels)
        w.writerows(summary_rows)


# ---------------------------------------------------------------------------
# LaTeX / console table
# ---------------------------------------------------------------------------

def print_latex_table(split_name: str, summary_rows: list,
                      col_labels: list, tags: list, rows_for_n: list):
    col_w = 14
    n_sys = len(tags)

    ns = [sum(1 for r in rows_for_n if r.get(f"{tag}_psnr") is not None) for tag in tags]
    ns_str = "  ".join(f"{lbl} n={n}" for lbl, n in zip(col_labels, ns))

    header = f"{'Metric':<22}" + "".join(f"  {lbl:>{col_w}}" for lbl in col_labels)
    sep = "=" * (22 + (col_w + 2) * n_sys)

    print(f"\n{'='*60}")
    print(f"  [{split_name.upper()}]  {ns_str}")
    print(sep)
    print(header)
    print(sep)

    row_idx = 0
    for group_name, metrics in METRIC_GROUPS:
        print(f"\n--- {group_name} ---")
        for label, _, _, higher_is_better in metrics:
            srow   = summary_rows[row_idx]
            strs   = srow[1:]
            row_idx += 1

            try:
                floats = [float(s) for s in strs]
                best   = max(floats) if higher_is_better else min(floats)
                disp   = [f"**{s}**" if abs(float(s) - best) < 1e-9 else s
                          for s in strs]
            except ValueError:
                disp = strs

            print(f"  {label:<20}" + "".join(f"  {d:>{col_w}}" for d in disp))

    print(sep)

    # LaTeX tabular
    col_spec = "l" + "c" * n_sys
    print(f"\n% LaTeX source — {split_name}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")
    print("Metric & " + " & ".join(col_labels) + " \\\\")
    print("\\hline")

    row_idx = 0
    for group_name, metrics in METRIC_GROUPS:
        print(f"\\multicolumn{{{n_sys + 1}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for label, _, _, higher_is_better in metrics:
            srow   = summary_rows[row_idx]
            strs   = srow[1:]
            row_idx += 1
            try:
                floats = [float(s) for s in strs]
                best   = max(floats) if higher_is_better else min(floats)
                tex    = [f"\\textbf{{{s}}}" if abs(float(s) - best) < 1e-9 else s
                          for s in strs]
            except ValueError:
                tex = strs
            clean = label.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
            print(f"  {clean} & " + " & ".join(tex) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")


# ---------------------------------------------------------------------------
# FID buffer merge helper
# ---------------------------------------------------------------------------

def merge_fid_bufs(a: dict, b: dict, tag_map: dict) -> dict:
    """
    Merge two fid_bufs dicts using tag_map = {new_tag: old_tag}.
    Returns a new fid_bufs dict keyed by new_tag.
    """
    merged = {}
    for new_tag, old_tag in tag_map.items():
        merged[new_tag] = {
            k: a.get(old_tag, {}).get(k, []) + b.get(old_tag, {}).get(k, [])
            for k in ("hair_real", "hair_fake", "bnd_real", "bnd_fake")
        }
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    M.try_load_face_model()

    split_results = {}  # split_name -> (rows, fid_bufs)

    for split_name, cfg in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"  Split: {split_name.upper()}")

        stems = discover_stems(cfg)
        tags  = [t for t, _, _ in cfg["systems"]]
        print(f"  교집합 stem 수: {len(stems)}  |  systems: {tags}")
        if len(stems) < 500:
            print(f"  [WARNING] FID는 500장 이상 권장 (현재 {len(stems)}장) — 참고용")

        rows, fid_bufs = evaluate_split(split_name, cfg, stems)

        print(f"\n  FID 계산 중 ({split_name})...")
        fid_results = compute_fid_for_split(fid_bufs)
        for k, v in fid_results.items():
            print(f"    {k}: {fmt(v)}")

        summary = aggregate_summary(rows, fid_results, tags)
        write_per_image_csv(rows, OUT_DIR / f"{split_name}_per_image.csv")
        write_summary_csv(summary, OUT_DIR / f"{split_name}_summary.csv", cfg["col_labels"])
        print_latex_table(split_name, summary, cfg["col_labels"], tags, rows)

        split_results[split_name] = (rows, fid_bufs)

    # ------------------------------------------------------------------ #
    # Combined: braid(dit_v4) + unbraid(ours)  →  unified 2-col table    #
    # Rename braid's "dit_v4" → "ours", keep unbraid's "ours" as-is      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  Split: COMBINED  (braid/dit_v4 + unbraid/ours)")

    braid_rows, braid_fid   = split_results["braid"]
    unbraid_rows, unbraid_fid = split_results["unbraid"]

    # Remap braid rows: rename dit_v4_* → ours_*
    def remap_row(r: dict) -> dict:
        new = {}
        for k, v in r.items():
            new[k.replace("dit_v4_", "ours_")] = v
        return new

    combined_rows = [remap_row(r) for r in braid_rows] + list(unbraid_rows)

    # Merge FID buffers: braid/dit_v4 + unbraid/ours → combined/ours
    combined_fid = {
        "shs": {
            k: braid_fid["shs"][k] + unbraid_fid["shs"][k]
            for k in ("hair_real", "hair_fake", "bnd_real", "bnd_fake")
        },
        "ours": {
            k: braid_fid["dit_v4"][k] + unbraid_fid["ours"][k]
            for k in ("hair_real", "hair_fake", "bnd_real", "bnd_fake")
        },
    }

    print(f"  전체 stem 수: {len(combined_rows)}")
    if len(combined_rows) < 500:
        print(f"  [WARNING] FID는 500장 이상 권장 (현재 {len(combined_rows)}장) — 참고용")

    print("\n  FID 계산 중 (combined)...")
    fid_combined = compute_fid_for_split(combined_fid)
    for k, v in fid_combined.items():
        print(f"    {k}: {fmt(v)}")

    combined_summary = aggregate_summary(combined_rows, fid_combined, list(COMBINED_TAGS))
    write_per_image_csv(combined_rows, OUT_DIR / "combined_per_image.csv")
    write_summary_csv(combined_summary, OUT_DIR / "combined_summary.csv", COMBINED_LABELS)
    print_latex_table("combined", combined_summary, COMBINED_LABELS,
                      list(COMBINED_TAGS), combined_rows)

    print(f"\n{'='*60}")
    print("저장 완료:")
    for name in ["braid", "unbraid", "combined"]:
        print(f"  eval_results/{name}_summary.csv")
        print(f"  eval_results/{name}_per_image.csv")


if __name__ == "__main__":
    main()
