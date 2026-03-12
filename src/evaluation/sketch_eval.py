"""
SketchEvaluator: orchestrates all metrics for a dataset split.

Computes per-sample: SHR, MCS, LPIPS (within matte), and optionally BSS.
Produces aggregate statistics and JSON report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import lpips as lpips_lib
import torch

from .metrics import compute_shr, compute_mcs, compute_bss


class SketchEvaluator:
    """
    Args:
        device:   torch device
        is_braid: if True, also compute BSS (braid structure score)
    """

    def __init__(self, device: torch.device, is_braid: bool = False):
        self.device = device
        self.is_braid = is_braid
        self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_batch(
        self,
        pred_rgb: torch.Tensor,    # (B, 3, H, W) [0, 1]
        target_rgb: torch.Tensor,  # (B, 3, H, W) [0, 1]
        sketch: torch.Tensor,      # (B, 3, H, W) [0, 1]
        matte: torch.Tensor,       # (B, 1, H, W) [0, 1]
        filenames: list[str],
    ) -> list[dict]:
        """
        Compute all metrics for a batch.

        Returns:
            list of dicts, one per sample, with keys: filename, shr, mcs, lpips, [bss]
        """
        B = pred_rgb.shape[0]

        shr = compute_shr(pred_rgb, sketch, matte)        # (B,)
        mcs = compute_mcs(pred_rgb, matte)                # (B,)

        # LPIPS within matte region
        pred_11   = pred_rgb   * 2.0 - 1.0
        target_11 = target_rgb * 2.0 - 1.0
        lpips_val = self.lpips_fn(
            pred_11 * matte,
            target_11 * matte,
        ).squeeze()  # (B,) or scalar

        if lpips_val.dim() == 0:
            lpips_val = lpips_val.unsqueeze(0).expand(B)

        bss = compute_bss(pred_rgb, sketch, matte) if self.is_braid else [None] * B

        results = []
        for i in range(B):
            r = {
                "filename": filenames[i],
                "shr":      shr[i].item(),
                "mcs":      mcs[i].item(),
                "lpips":    lpips_val[i].item(),
            }
            if self.is_braid:
                r["bss"] = bss[i].item()
            results.append(r)

        return results

    def save_report(self, results: list[dict], output_path: Path):
        """Save per-sample metrics and aggregate stats to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self._aggregate(results)
        report = {"summary": summary, "per_sample": results}

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def print_summary(self, results: list[dict]):
        summary = self._aggregate(results)
        print("\n=== Evaluation Summary ===")
        for k, v in summary.items():
            print(f"  {k:10s}: {v:.4f}")

    def _aggregate(self, results: list[dict]) -> dict:
        keys = [k for k in results[0].keys() if k != "filename"]
        summary = {}
        for k in keys:
            vals = [r[k] for r in results if r[k] is not None]
            if vals:
                summary[f"{k}_mean"] = sum(vals) / len(vals)
                summary[f"{k}_std"]  = float(torch.tensor(vals).std().item())
        return summary
