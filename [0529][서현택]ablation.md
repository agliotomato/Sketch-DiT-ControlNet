### Contribution points
- Hair-Specific Matte-Gated ControlNet Residual Schedule (architectural)
- SOTA Sketch-to-Hair Editing (empirical )
- Curriculum-Trained Unified Model for Multi-Category Hair Editing (empirical)

### ablation 실험 정리

1. Curriculum Training (unbraid→braid) 타당성 증명

- 현재 방향 : Curriculum Training 결과로 braid/unbraid 양쪽 평가
- unbraid-only (3000장) → unbraid 결과 뽑기 (O)
- braid-only (1000장) → braid 결과 뽑기
- Reverse Curriculum (braid→unbraid) → braid/unbraid 결과 뽑기
- Joint Training (braid+unbraid, 4000장) → braid/unbraid 결과 뽑기
- **Curriculum Training (unbraid→braid, ours)** → braid/unbraid 결과 뽑기 (O)
