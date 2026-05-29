#!/bin/bash
set -e

# C1 — Unbraid Only
echo "===== C1: Unbraid Only ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/phase1_unbraid.yaml \
  --checkpoint checkpoints/phase1_unbraid/epoch_40.pth \
  --split unbraid_test \
  --output_dir custom_results/ablation_cl_run2/c1_unbraid_only/unbraid

# C5 — Ours (Forward Curriculum)
echo "===== C5: Ours — unbraid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/phase2_braid.yaml \
  --checkpoint checkpoints/phase2_braid/final.pth \
  --split unbraid_test \
  --output_dir custom_results/ablation_cl_run2/c5_ours/unbraid

echo "===== C5: Ours — braid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/phase2_braid.yaml \
  --checkpoint checkpoints/phase2_braid/final.pth \
  --split braid_test \
  --output_dir custom_results/ablation_cl_run2/c5_ours/braid

echo "===== C1 + C5 완료 ====="
