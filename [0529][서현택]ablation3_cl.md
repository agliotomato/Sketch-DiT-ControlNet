## Ablation 3. Curriculum Learning

**검증 대상**: unbraid → braid 순서(easy-to-hard)가 최적인가.

| ID | 학습 방식 | 설명 |
|----|----------|------|
| C1 | Unbraid Only | unbraid 3K, braid 미학습 |
| C2 | Braid Only | braid 1K, unbraid 없이 처음부터 학습 |
| C3 | Joint | unbraid + braid 동시 (ConcatDataset, 4K) |
| C4 | Reverse Curriculum | braid pretrain → unbraid finetune |
| **C5 (Ours)** | Forward Curriculum | unbraid pretrain → braid finetune |

**검증 포인트**:
- C2 vs C5 (braid 성능): curriculum이 없으면 1K 소량 데이터로 braid 수렴 어려움
- C3 vs C5 (braid 성능): 3:1 데이터 불균형이 braid 학습 속도 저하시킴
- C4 vs C5: easy-to-hard 순서 자체가 결정적임을 증명
- C1 vs C5 (unbraid 성능): braid fine-tuning 이후 catastrophic forgetting 없음을 증명

**평가 방식**: 생성된 hair patch에 대해서만 평가. Boundary LPIPS·Face LPIPS·ArcFace cosine은 제외 — 경계 품질과 얼굴 동일성 보존은 Matte-Gated ControlNet 아키텍처가 결정하는 요소이지, 커리큘럼 학습 순서가 결정하는 요소가 아니기 때문.

**측정 지표** (unbraid / braid 평가셋 각각):
- Hair FID, Hair LPIPS, SSIM, PSNR
- Edge IoU, Chamfer Distance (braid 구조 정밀도)

