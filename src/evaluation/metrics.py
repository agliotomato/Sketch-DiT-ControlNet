"""
Evaluation metrics for sketch-conditioned hair generation.

Metrics:
  SHR  - Sketch Stroke Hit Rate         (primary: does generated hair follow sketch?)
  MCS  - Matte Containment Score        (is generated hair within matte region?)
  BSS  - Braid Structure Score          (braid-specific: strand crossing accuracy)
  LPIPS within matte                    (perceptual quality inside hair region)
  FID on matte bbox crops               (distribution quality)
"""

from __future__ import annotations

import kornia.filters as KF
import kornia.morphology as KM
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SHR: Sketch Stroke Hit Rate
# ---------------------------------------------------------------------------

def compute_shr(
    pred_rgb: torch.Tensor,
    sketch: torch.Tensor,
    matte: torch.Tensor,
    stroke_threshold: float = 0.1,
    edge_threshold_percentile: float = 75.0,
) -> torch.Tensor:
    """
    SHR = (sketch_strokes AND pred_edges).sum() / sketch_strokes.sum()

    High SHR: where the user drew a stroke, the model generated a strand edge.

    Args:
        pred_rgb:   (B, 3, H, W) [0, 1]
        sketch:     (B, 3, H, W) [0, 1]
        matte:      (B, 1, H, W) [0, 1]
        stroke_threshold: threshold for detecting sketch strokes
        edge_threshold_percentile: percentile threshold for binarizing predicted edges

    Returns:
        shr: (B,) per-sample SHR values
    """
    B = pred_rgb.shape[0]

    # Sketch stroke mask (where any channel > threshold, inside matte)
    sketch_mask = (sketch.max(dim=1, keepdim=True).values > stroke_threshold).float()
    sketch_mask = sketch_mask * matte  # only count strokes inside matte

    # Sobel edge map from generated hair
    pred_gray = pred_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    grad = KF.spatial_gradient(pred_gray)           # (B, 1, 2, H, W)
    edge_mag = grad.norm(dim=2)                     # (B, 1, H, W)

    # Adaptive threshold per image
    shr_values = []
    for i in range(B):
        sketch_i = sketch_mask[i, 0]  # (H, W)
        edge_i = edge_mag[i, 0]       # (H, W)

        # Threshold edges at given percentile (only within matte)
        matte_i = matte[i, 0]
        edge_vals = edge_i[matte_i > 0.1]
        if edge_vals.numel() == 0:
            shr_values.append(torch.tensor(0.0))
            continue

        threshold = torch.quantile(edge_vals, edge_threshold_percentile / 100.0)
        edge_bin = (edge_i > threshold).float()

        n_strokes = sketch_i.sum()
        if n_strokes == 0:
            shr_values.append(torch.tensor(1.0))
            continue

        hits = (sketch_i * edge_bin).sum()
        shr_values.append(hits / n_strokes)

    return torch.stack(shr_values)


# ---------------------------------------------------------------------------
# MCS: Matte Containment Score
# ---------------------------------------------------------------------------

def compute_mcs(
    pred_rgb: torch.Tensor,
    matte: torch.Tensor,
    content_threshold: float = 0.05,
) -> torch.Tensor:
    """
    MCS = (pred_content AND matte > 0.1).sum() / pred_content.sum()

    Measures: what fraction of generated hair pixels are within the matte region.

    Args:
        pred_rgb:          (B, 3, H, W) [0, 1]
        matte:             (B, 1, H, W) [0, 1]
        content_threshold: threshold to detect non-black (hair) pixels

    Returns:
        mcs: (B,) per-sample MCS values
    """
    # Content mask: pixels where model generated something (not black)
    content_mask = (pred_rgb.max(dim=1, keepdim=True).values > content_threshold).float()

    matte_bin = (matte > 0.1).float()
    inside_matte = (content_mask * matte_bin).sum(dim=[1, 2, 3])
    total_content = content_mask.sum(dim=[1, 2, 3]).clamp(min=1)

    return inside_matte / total_content


# ---------------------------------------------------------------------------
# BSS: Braid Structure Score (braid fine-tune specific)
# ---------------------------------------------------------------------------

def _count_crossings(edge_map: torch.Tensor) -> float:
    """
    Approximate X-junction count in an edge map.
    X-junctions indicate strand crossings in braid images.

    Heuristic: convolve with a cross-shaped kernel and count strong responses.
    """
    # Dilate edge map slightly for robustness
    kernel = torch.ones(3, 3, device=edge_map.device)
    edge_dilated = KM.dilation(edge_map.unsqueeze(0).unsqueeze(0), kernel).squeeze()

    # X-junction detector: 4 directions must all be active
    # Approximate: local maximum in a neighborhood
    edge_f = edge_dilated.unsqueeze(0).unsqueeze(0)
    local_max = F.max_pool2d(edge_f, kernel_size=5, stride=1, padding=2)
    x_junctions = (local_max > 0.5).float().sum().item()
    return x_junctions


def compute_bss(
    pred_rgb: torch.Tensor,
    sketch: torch.Tensor,
    matte: torch.Tensor,
    edge_threshold: float = 0.15,
) -> torch.Tensor:
    """
    BSS = |crossings_pred - crossings_sketch| / crossings_sketch

    Lower = better braid topology reproduction.

    Args:
        pred_rgb: (B, 3, H, W) [0, 1]
        sketch:   (B, 3, H, W) [0, 1]
        matte:    (B, 1, H, W) [0, 1]

    Returns:
        bss: (B,) per-sample BSS values
    """
    B = pred_rgb.shape[0]
    bss_values = []

    pred_gray = pred_rgb.mean(dim=1, keepdim=True)
    grad = KF.spatial_gradient(pred_gray)
    edge_mag = grad.norm(dim=2)  # (B, 1, H, W)

    sketch_gray = sketch.mean(dim=1, keepdim=True)
    sketch_grad = KF.spatial_gradient(sketch_gray)
    sketch_edge = sketch_grad.norm(dim=2)

    for i in range(B):
        m = matte[i, 0]
        pred_edge = (edge_mag[i, 0] > edge_threshold).float() * (m > 0.1).float()
        sk_edge   = (sketch_edge[i, 0] > edge_threshold).float() * (m > 0.1).float()

        c_pred = _count_crossings(pred_edge)
        c_sk   = _count_crossings(sk_edge)

        if c_sk == 0:
            bss_values.append(torch.tensor(0.0))
        else:
            bss_values.append(torch.tensor(abs(c_pred - c_sk) / c_sk))

    return torch.stack(bss_values)
