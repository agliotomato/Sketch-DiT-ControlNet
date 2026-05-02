"""
Curriculum learning 3-way ablation: 비교 표 + loss curve 시각화

Usage:
  python scripts/compare_curriculum.py \
    --results_dirs eval_results/ablation/phase1_only \
                   eval_results/ablation/phase2_scratch \
                   eval_results/ablation/phase2_curriculum \
    --labels "Phase1 Only" "Phase2 Scratch" "Curriculum (ours)" \
    --ckpt_dirs checkpoints/phase1_unbraid \
                checkpoints/phase2_braid_scratch \
                checkpoints/phase2_braid \
    --output_dir eval_results/ablation/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def load_metrics(result_dir: Path) -> dict | None:
    p = result_dir / "metrics.json"
    if not p.exists():
        print(f"  [WARN] metrics.json not found: {p}")
        return None
    with open(p) as f:
        data = json.load(f)
    return data["summary"]


def print_markdown_table(rows: list[dict], labels: list[str]) -> None:
    header = "| Variant | BSS↓ | SHR↑ | MCS↑ | LPIPS↓ |"
    sep    = "|---------|------|------|------|--------|"
    print("\n## 정량 비교 (braid_test)")
    print(header)
    print(sep)
    for label, row in zip(labels, rows):
        if row is None:
            print(f"| {label} | — | — | — | — |")
        else:
            print(
                f"| {label} "
                f"| {row['bss']:.4f} "
                f"| {row['shr']:.4f} "
                f"| {row['mcs']:.4f} "
                f"| {row['lpips']:.4f} |"
            )
    print()


def print_latex_table(rows: list[dict], labels: list[str]) -> None:
    print("## LaTeX table")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Variant & BSS$\downarrow$ & SHR$\uparrow$ & MCS$\uparrow$ & LPIPS$\downarrow$ \\")
    print(r"\midrule")
    for label, row in zip(labels, rows):
        if row is None:
            print(f"{label} & — & — & — & — \\\\")
        else:
            print(
                f"{label} & {row['bss']:.4f} & {row['shr']:.4f} "
                f"& {row['mcs']:.4f} & {row['lpips']:.4f} \\\\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


# ---------------------------------------------------------------------------
# TensorBoard loss curves
# ---------------------------------------------------------------------------

def find_event_file(base_dir: Path) -> Path | None:
    """Accelerate stores TB events in {output_dir}/logs/<tracker_name>/events.*"""
    for p in sorted(base_dir.rglob("events.out.tfevents.*")):
        return p
    return None


def read_tb_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    """Returns (steps, values) for the given scalar tag."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [WARN] tensorboard package not found; skipping loss curves")
        return [], []

    ea = EventAccumulator(str(event_file))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        print(f"  [WARN] tag '{tag}' not found in {event_file}. Available: {available}")
        return [], []

    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values


def smooth(values: list[float], alpha: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    out = []
    v = values[0] if values else 0.0
    for x in values:
        v = alpha * v + (1 - alpha) * x
        out.append(v)
    return out


def plot_loss_curves(
    ckpt_dirs: list[Path],
    labels: list[str],
    output_path: Path,
    tag: str = "loss_total",
) -> bool:
    colors = ["#e07b39", "#4c72b0", "#55a868"]  # orange, blue, green
    fig, ax = plt.subplots(figsize=(8, 4.5))

    any_plotted = False
    for ckpt_dir, label, color in zip(ckpt_dirs, labels, colors):
        event_file = find_event_file(ckpt_dir)
        if event_file is None:
            print(f"  [WARN] No TensorBoard event file in {ckpt_dir}")
            continue

        steps, values = read_tb_scalars(event_file, tag)
        if not steps:
            continue

        steps_arr  = np.array(steps)
        values_arr = np.array(values, dtype=float)
        smoothed   = np.array(smooth(values_arr.tolist(), alpha=0.92))

        ax.plot(steps_arr, values_arr, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(steps_arr, smoothed,   color=color, linewidth=2.0, label=label)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Loss (flow + lpips + edge)", fontsize=12)
    ax.set_title("Curriculum Ablation: Training Loss Curves", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved: {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dirs", nargs="+", required=True,
        help="각 variant의 evaluate.py output_dir (metrics.json 위치)",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="variant 이름 (results_dirs와 순서 동일)",
    )
    parser.add_argument(
        "--ckpt_dirs", nargs="+", default=None,
        help="TensorBoard event 파일 탐색용 checkpoint 디렉토리 (선택, results_dirs와 순서 동일)",
    )
    parser.add_argument(
        "--output_dir", default="eval_results/ablation/",
        help="비교 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--tb_tag", default="loss_total",
        help="읽을 TensorBoard scalar 태그 (기본: loss_total)",
    )
    args = parser.parse_args()

    assert len(args.results_dirs) == len(args.labels), \
        "--results_dirs 와 --labels 개수가 달라요"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 메트릭 표 ---
    result_dirs = [Path(d) for d in args.results_dirs]
    rows = [load_metrics(d) for d in result_dirs]

    print_markdown_table(rows, args.labels)
    print_latex_table(rows, args.labels)

    # --- 2. 메트릭 JSON 저장 ---
    summary = {}
    for label, row in zip(args.labels, rows):
        summary[label] = row if row is not None else {}
    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison JSON saved: {output_dir / 'comparison_metrics.json'}")

    # --- 3. Loss curve ---
    if args.ckpt_dirs:
        assert len(args.ckpt_dirs) == len(args.labels), \
            "--ckpt_dirs 와 --labels 개수가 달라요"
        ckpt_dirs = [Path(d) for d in args.ckpt_dirs]
        plot_loss_curves(
            ckpt_dirs=ckpt_dirs,
            labels=args.labels,
            output_path=output_dir / "loss_curves.png",
            tag=args.tb_tag,
        )
    else:
        print("--ckpt_dirs 미지정: loss curve 생략")


if __name__ == "__main__":
    main()
