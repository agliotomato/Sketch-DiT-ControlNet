# contribution points
- Hair-Specific Matte-Gated ControlNet Residual Schedule (architectural)
- SOTA Sketch-to-Hair Editing (empirical )
- Curriculum-Trained Unified Model for Multi-Category Hair Editing (empirical)

---
## Ablation A. Matte Conditioning 설계

**검증 대상**: MatteCNN과 matte_down 각각이 hair-only 생성에 기여하는가.

현재 구조:
```
sketch → VAE.encode → sketch_latent  (B, 16, 64, 64)
matte  → MatteCNN   → matte_feat    (B, 16, 64, 64)
matte  → bilinear downsample        (B,  1, 64, 64)  ← matte_down
ctrl_cond = cat([sketch_latent + matte_feat, matte_down], dim=1)  # (B, 17, 64, 64)
```

| ID | 변형 | ctrl_cond 구성 | 증명 목적 |
|----|------|---------------|----------|
| A1 | matte_down만 (MatteCNN 없음) | `cat([sketch_latent, matte_down])` | MatteCNN 필요성 |
| A2 | MatteCNN만 (matte_down 없음) | `cat([sketch_latent + matte_feat, zeros_1ch])` | 명시적 공간 마스크 필요성 |
| **A3 (Ours)** | MatteCNN + matte_down | `cat([sketch_latent + matte_feat, matte_down])` | — |

- **A1 vs A3**: MatteCNN이 단순 다운샘플 마스크 위에 추가로 기여하는지 증명
- **A2 vs A3**: 명시적 matte_down가 MatteCNN 임베딩 위에 추가로 기여하는지 증명

**측정 지표**:
- Face LPIPS (hair 영역 외 보존도)
- ArcFace cosine similarity (identity 보존)

---

## Ablation B. Loss 구성

### 1. Edge Loss 기여 (braid finetune)

**검증 대상**: braid finetune 단계에서 edge loss가 strand 구조 정밀도에 기여하는가.

phase1_unbraid 체크포인트를 공유하고 braid finetune 손실만 다르게 적용 → clean 비교.

| 변형 | edge loss | 비고 |
|------|:---------:|------|
| Edge loss 없음 | X | `w_edge=0.0` |
| **Edge loss 있음 (Ours)** | O | `w_edge=0.05` (phase2_braid, 완료) |

**측정 지표** (braid 평가셋):
- Edge IoU, Chamfer Distance (strand 구조 정밀도)
- Hair LPIPS, PSNR (전반적 품질 변화 확인)

---

### 2. outside_weight 기여 (FlowMatchingLoss)

**검증 대상**: hair 영역 밖에 약한 supervision을 주는 것이 경계 품질에 기여하는가.

현재 구조:
```python
weight = matte_down + outside_weight * (1.0 - matte_down)
```

| 변형 | outside_weight | 의도 |
|------|:--------------:|------|
| 외부 완전 마스킹 | 0.0 | 경계 아티팩트 유발 예상 |
| **약한 외부 감독 (Ours)** | 0.1 | 경계 안정성 확보 |
| 강한 외부 감독 | 0.5 | 과도한 supervision |

**측정 지표**:
- Boundary LPIPS (경계 품질)
- Face LPIPS (외부 감독 강도에 따른 얼굴 영역 영향 확인)

---

## Ablation C. Curriculum Learning

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

---

## SOTA 비교 (Ablation이 아닌 Comparison)

**입력 스케치**: StrokeColorSampler 적용 스케치 (stroke 색 = 실제 헤어 색). GT와 pixel-level 비교 가능, SHS와 동일 포맷으로 공정 비교.

> 임의 색 스트록은 정량 비교 대상이 아닌 qualitative 데모용으로 별도 처리. (GT 헤어 색과 매칭 불가 → LPIPS/SSIM/PSNR 측정 불가)

| 모델 | 방식 | 비고 |
|------|------|------|
| SketchHairSalon (SHS) | GAN-based | 기존 SOTA |
| SD3.5 + Vanilla ControlNet | standard ControlNet | matte gating 없음 |
| **Ours** | Matte-Gated ControlNet + curriculum | — |

**측정 지표**:
- Sketch 일치도: Edge IoU, Chamfer Distance
- 텍스처 품질: Hair FID
- 원본 재구성: LPIPS, SSIM, PSNR

---

## 실험 우선순위

| 우선순위 | Ablation | 이유 |
|:--------:|----------|------|
| 1 | **C (Curriculum)** | 완료 |
| 2 | **A (Matte)** | MatteCNN, matte_down 필요성, contribution 직결 |
| 3 | **B (Loss)** | phase1 체크포인트 재활용, 실험 비용 낮음 |

계획 
- 실험 C : 5월 29일
- 실험 A : 5월 30일
- 실험 B : 5월 31일
- SOTA 비교 : 5월 30일 ~ 31일

Ablation A
- matte conditiong 이 약하다면(A1, A2) ControlNet이 hair 영역 토큰 외에도 non-hair 토큰에도 영향을 줄 가능성이 있다고 생각하였습니다. 다른 조건의 matte-conditoing에 따른 영향을 확인하기 위해서 non-hair 영역도 평가지표에 포함시켰습ㅂ니다. 

- matte_conditong 의 타당성을 증명하는 평가는 헤어패치만 평가하려고 계획하였습니다. full-image 생성 원리가 헤어 패치 생성 후 latent space 내에서 헤어와 배경이 합성되기 때문에 생성 능력 평가는 헤어 패치만으로도 충분하다고 판단하였습니다. 그리하여 matte-conditiong 에서는 bounday 관련 평가지표는 제외하였습니다.

Ablation B
- 넵 알겠습니다