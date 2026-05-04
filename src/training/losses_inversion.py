"""
Inversion Adapter Loss Functions.

Phase A (direct supervision):
  L_structure = BCE(stroke_mask_pred, stroke_mask_GT)
  L_color     = L1(sketch_pred, sketch_GT) * stroke_mask_GT  (stroke 영역만)
  L_matte     = BCE(matte_pred, matte_GT) + L1(matte_pred, matte_GT)

Phase B (+ feature cycle consistency, phase_b_start epoch 이후 추가):
  L_feature   = mean MSE(block_pred[i], block_gt[i]) over 12 blocks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InversionLoss(nn.Module):
    def __init__(
        self,
        w_structure: float = 1.0,
        w_color: float = 0.5,
        w_matte: float = 1.0,
        w_feature: float = 0.1,
    ):
        super().__init__()
        self.w_structure = w_structure
        self.w_color     = w_color
        self.w_matte     = w_matte
        self.w_feature   = w_feature

    def forward(
        self,
        stroke_mask_pred: torch.Tensor,   # (B, 1, 512, 512)
        sketch_pred: torch.Tensor,         # (B, 3, 512, 512)
        matte_pred: torch.Tensor,          # (B, 1, 512, 512)
        sketch_gt: torch.Tensor,           # (B, 3, 512, 512)
        matte_gt: torch.Tensor,            # (B, 1, 512, 512)
        block_pred: list[torch.Tensor] | None = None,   # Phase B
        block_gt:   list[torch.Tensor] | None = None,
        w_feature_current: float | None = None,         # warm-up된 동적 가중치
    ) -> tuple[torch.Tensor, dict]:

        # bfloat16 → float32 (BCE는 float32만 지원)
        stroke_mask_pred = stroke_mask_pred.float()
        sketch_pred      = sketch_pred.float()
        matte_pred       = matte_pred.float()
        sketch_gt        = sketch_gt.float()
        matte_gt         = matte_gt.float()

        # stroke_mask_GT: sketch_GT에서 비배경(non-black) 픽셀
        stroke_mask_gt = (sketch_gt.max(dim=1, keepdim=True).values > 0.05).float()

        # --- Phase A losses ---
        l_structure = F.binary_cross_entropy(stroke_mask_pred, stroke_mask_gt)

        # L_color: stroke 영역 픽셀 수로 나눠 sparse mask 희석 방지
        n_stroke = stroke_mask_gt.sum().clamp(min=1)
        l_color = (F.l1_loss(sketch_pred, sketch_gt, reduction="none") * stroke_mask_gt).sum() / n_stroke

        # Matte loss: BCE + L1 + Dice
        # 입력이 전체 사진이므로 배경→0을 강하게 밀어내야 함
        # Dice loss: 클래스 불균형(배경 多, 헤어 少)에 강건
        l_matte_bce  = F.binary_cross_entropy(matte_pred, matte_gt)
        l_matte_l1   = F.l1_loss(matte_pred, matte_gt)
        smooth       = 1e-6
        intersection = (matte_pred * matte_gt).sum(dim=[1, 2, 3])
        union        = matte_pred.sum(dim=[1, 2, 3]) + matte_gt.sum(dim=[1, 2, 3])
        l_matte_dice = (1 - (2 * intersection + smooth) / (union + smooth)).mean()
        l_matte      = l_matte_bce + l_matte_l1 + l_matte_dice

        total = (
            self.w_structure * l_structure
            + self.w_color   * l_color
            + self.w_matte   * l_matte
        )
        log = {
            "loss_structure": l_structure.item(),
            "loss_color":     l_color.item(),
            "loss_matte":     l_matte.item(),
        }

        # --- Phase B: feature cycle consistency (w_feature_current로 동적 조절) ---
        if block_pred is not None and block_gt is not None:
            l_feature = sum(
                F.mse_loss(bp.float(), bg.float())
                for bp, bg in zip(block_pred, block_gt)
            ) / len(block_pred)
            # w_feature_current: trainer가 warm-up 스케줄 반영해 전달
            w = w_feature_current if w_feature_current is not None else self.w_feature
            total = total + w * l_feature
            log["loss_feature"]         = l_feature.item()
            log["w_feature_current"]    = w

        log["loss_total"] = total.item()
        return total, log


def tv_loss(mask: torch.Tensor) -> torch.Tensor:
    """Total Variation loss: penalizes abrupt mask discontinuities → smoother strokes."""
    diff_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
    diff_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w
