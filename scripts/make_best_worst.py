"""
Best/Worst 20 visualization generator.

Ranking criterion: ΔSSIM = ours_ssim - shs_ssim  (per image)
  Best 20  : largest ΔSSIM  (DiT v4 improves most over SHS)
  Worst 20 : smallest ΔSSIM (DiT v4 regresses or improves least vs SHS)

Input : eval_results/combined_per_image.csv
Output: best_worst.md
"""

import csv
import math
from pathlib import Path

ROOT     = Path(__file__).parent.parent
CSV_PATH = ROOT / "eval_results" / "combined_per_image.csv"
OUT_PATH = ROOT / "best_worst.md"


def img_path(stem: str, split: str, kind: str) -> str:
    if split == "braid":
        paths = {
            "sketch": f"dataset/braid/sketch/test/{stem}.png",
            "gt":     f"dataset/braid/img/test/{stem}.png",
            "shs":    f"custom_results/gan/shs/{stem}.png",
            "ours":   f"custom_results/dit/weighted_sum_v4/{stem}_full.png",
        }
    else:
        paths = {
            "sketch": f"dataset/unbraid/sketch/test/{stem}.png",
            "gt":     f"dataset/unbraid/img/test/{stem}.png",
            "shs":    f"custom_results/gan/generated_unbraid/{stem}.png",
            "ours":   f"custom_results/dit/inferunbraidbybraidckp/{stem}_full.png",
        }
    return paths[kind]


def safe_float(v):
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            s = safe_float(r.get("shs_ssim"))
            o = safe_float(r.get("ours_ssim"))
            if s is None or o is None:
                continue
            rows.append({
                "stem":       r["stem"],
                "split":      r["split"],
                "delta_ssim": o - s,
                "shs_ssim":   s,
                "ours_ssim":  o,
                "shs_psnr":   safe_float(r.get("shs_psnr")),
                "ours_psnr":  safe_float(r.get("ours_psnr")),
                "shs_lpips":  safe_float(r.get("shs_lpips")),
                "ours_lpips": safe_float(r.get("ours_lpips")),
            })
    return rows


def fmt_delta(v):
    return f"{v:+.4f}" if v is not None else "N/A"


def metric_cell(ssim, psnr, lpips) -> str:
    s = f"{ssim:.4f}" if ssim is not None else "N/A"
    p = f"{psnr:.2f}"  if psnr  is not None else "N/A"
    l = f"{lpips:.4f}" if lpips is not None else "N/A"
    return f"SSIM&nbsp;`{s}`<br/>PSNR&nbsp;`{p}`<br/>LPIPS&nbsp;`{l}`"


def make_table_rows(entries: list) -> str:
    lines = []
    lines.append("| # / Split / Stem / ΔSSIM | Sketch | GT | SHS | DiT v4 |")
    lines.append("|---|:---:|---|---|---|")
    for i, e in enumerate(entries, 1):
        stem  = e["stem"]
        split = e["split"]
        delta = fmt_delta(e["delta_ssim"])

        sk_p   = img_path(stem, split, "sketch")
        gt_p   = img_path(stem, split, "gt")
        shs_p  = img_path(stem, split, "shs")
        ours_p = img_path(stem, split, "ours")

        shs_m  = metric_cell(e["shs_ssim"],  e["shs_psnr"],  e["shs_lpips"])
        ours_m = metric_cell(e["ours_ssim"], e["ours_psnr"], e["ours_lpips"])

        info = f"**{i}** {split}<br/>`{stem}`<br/>Δ&nbsp;**{delta}**"

        lines.append(
            f"| {info} "
            f"| ![sk]({sk_p}) "
            f"| ![GT]({gt_p})<br/>SSIM&nbsp;`—`<br/>PSNR&nbsp;`—`<br/>LPIPS&nbsp;`—` "
            f"| ![SHS]({shs_p})<br/>{shs_m} "
            f"| ![DiT v4]({ours_p})<br/>{ours_m} |"
        )
    return "\n".join(lines)


def main():
    rows = load_rows()
    rows.sort(key=lambda x: x["delta_ssim"], reverse=True)

    best20  = rows[:20]
    worst20 = rows[-20:][::-1]

    braid_rows = sorted([r for r in rows if r["split"] == "braid"],
                        key=lambda x: x["delta_ssim"], reverse=True)
    braid_best5  = braid_rows[:5]
    braid_worst5 = braid_rows[-5:][::-1]

    n_braid   = len(braid_rows)
    n_unbraid = sum(1 for r in rows if r["split"] == "unbraid")

    lines = []
    lines.append(f"> 랭킹 기준: ΔSSIM = DiT v4 − SHS (per image) | 전체 {len(rows)}장 (braid {n_braid} + unbraid {n_unbraid})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Best 20 (braid + unbraid)")
    lines.append("")
    lines.append(make_table_rows(best20))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Worst 20 (braid + unbraid)")
    lines.append("")
    lines.append(make_table_rows(worst20))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Braid Best 5  (braid {n_braid}장 중)")
    lines.append("")
    lines.append(make_table_rows(braid_best5))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Braid Worst 5  (braid {n_braid}장 중)")
    lines.append("")
    lines.append(make_table_rows(braid_worst5))
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장 완료: {OUT_PATH}")
    print(f"  Best 20  ΔSSIM 범위: {best20[-1]['delta_ssim']:+.4f} ~ {best20[0]['delta_ssim']:+.4f}")
    print(f"  Worst 20 ΔSSIM 범위: {worst20[0]['delta_ssim']:+.4f} ~ {worst20[-1]['delta_ssim']:+.4f}")
    print(f"  Braid Best 5  ΔSSIM: {braid_best5[-1]['delta_ssim']:+.4f} ~ {braid_best5[0]['delta_ssim']:+.4f}")
    print(f"  Braid Worst 5 ΔSSIM: {braid_worst5[0]['delta_ssim']:+.4f} ~ {braid_worst5[-1]['delta_ssim']:+.4f}")


if __name__ == "__main__":
    main()
