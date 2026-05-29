#!/bin/bash
set -e

# C2 — Braid Only
echo "===== C2: Braid Only ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/ablation_braid_only.yaml \
  --checkpoint checkpoints/ablation_braid_only/final.pth \
  --split braid_test \
  --output_dir custom_results/ablation_cl_run2/c2_braid_only/braid

# C3 — Joint
echo "===== C3: Joint — unbraid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/ablation_joint.yaml \
  --checkpoint checkpoints/ablation_joint/final.pth \
  --split unbraid_test \
  --output_dir custom_results/ablation_cl_run2/c3_joint/unbraid

echo "===== C3: Joint — braid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/ablation_joint.yaml \
  --checkpoint checkpoints/ablation_joint/final.pth \
  --split braid_test \
  --output_dir custom_results/ablation_cl_run2/c3_joint/braid

# C4 — Reverse Curriculum
echo "===== C4: Reverse — unbraid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/ablation_reverse_curriculum.yaml \
  --checkpoint checkpoints/ablation_reverse_curriculum/final.pth \
  --split unbraid_test \
  --output_dir custom_results/ablation_cl_run2/c4_reverse/unbraid

echo "===== C4: Reverse — braid ====="
CUDA_VISIBLE_DEVICES=0 python scripts/infer_ablation_cl.py \
  --config configs/ablation_reverse_curriculum.yaml \
  --checkpoint checkpoints/ablation_reverse_curriculum/final.pth \
  --split braid_test \
  --output_dir custom_results/ablation_cl_run2/c4_reverse/braid

echo "===== C2 + C3 + C4 완료 ====="
