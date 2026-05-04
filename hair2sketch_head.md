# Hair → Sketch: Inverse Head 설계 및 구현

---

## 1. 목표

Hair photo → Sketch 역방향 생성.  
최종 목표: **Cyclic Supervised** — inverse가 만든 sketch를 forward가 다시 읽어 hair를 복원.

- **단방향 (sketch→hair)**: 수렴 달성. forward ControlNet이 "hair 의미 평가기" 역할을 할 수 있는 근거 확보.
- **핵심 실험 질문**: forward의 내부 feature를 고정된 평가 기준으로 삼아 역방향(hair→sketch)을 cycle-supervised로 학습했을 때, sketch 품질이 실제로 개선되는가?
- **확장 방향**: cycle이 유효하면 → paired data 없는 hair 이미지 추가 투입 (semi-supervised, 데이터 효율 클레임).

### 현재 단계

| 단계 | 상태 |
|---|---|
| Forward (sketch→hair) 학습 | ✅ 수렴 |
| Inverse (hair→sketch) 구현 | ✅ 완료 |
| **Inverse 학습 실행** | ⬅ 지금 여기 |
| Round-trip LPIPS 측정 | 대기 중 |
| Cycle 유무 ablation | 대기 중 |

---

## 2. 아키텍처 설계

### 핵심 원칙

- **VAE decoder 사용 안 함** — sketch는 이미지가 아니라 stroke mask에 가까움
- **Mask decoder** — stroke 구조만 예측, 색은 분리
- **Non-parametric color sampling** — 색 학습 파라미터 없음 (이유는 [한계 섹션](#61-non-parametric-color-sampling) 참고)

### 데이터 흐름

```
hair_image (B, 3, 512, 512)
  → frozen VAE encoder
  → hair_latent (B, 16, 64, 64)
  → frozen DiT + LoRA (sigma=0, null conditioning)
      hook @ all 24 blocks → (24, B, 1024, 1536)
      learnable agg_weights (3×24, softmax) → weighted sum per scale
      f_early (B, 1536, 32, 32)
      f_mid   (B, 1536, 32, 32)
      f_late  (B, 1536, 32, 32)
  → FPN Mask Decoder (UNet-style skip connection)
      32×32 → (cat f_mid) → 64 → (cat f_early) → 128 → 256 → 512×512
  → stroke_mask (B, 1, 512, 512)   [sigmoid]
  → matte_pred  (B, 1, 512, 512)   [sigmoid]
  → patch_color_sample(hair × matte_pred, stroke_mask)
  → sketch_pred (B, 3, 512, 512)
```

#### 단계별 설명

**① VAE encoder**

픽셀 공간 → latent 공간 압축. 512→64 (8배 다운), 3채널→16채널. frozen.

**② DiT 통과**

DiT는 원래 noisy latent + sigma + text를 받는 denoiser. 여기서는 clean image를 그대로 넣고(`sigma=0`), text는 null(빈 임베딩). forward pass를 feature 추출기로만 사용. LoRA가 이 OOD(Out-of-Distribution) 상황에서 attention을 조정.

**③ 중간 feature hook + 학습 가능 집계**

DiT의 최종 출력(denoised latent)은 필요 없다. VAE decoder로 복원하면 이미지가 나오는데 그건 이미지 생성이지 우리가 원하는 게 아님.

우리가 원하는 것은 hair 이미지의 **구조 정보** — 선이 어디 있는지, 머리카락이 어디 있는지. 그 정보는 DiT가 내부 블록을 통과하면서 만드는 중간 표현 안에 있다. hook은 그 중간 표현을 통과 도중에 빼내는 것.

```
hair_latent
  → block 0  → [표현]
  → block 1  → [표현]
  ...
  → block 23 → [최종 출력] ← 사용 안 함
         ↑
  각 블록 출력을 hook으로 캡처 → 구조 정보 추출에 사용
```

24블록 전부에서 token sequence `(B, 1024, 1536)`을 캡처. 학습 가능한 3×24 가중합 행렬(`agg_weights`, softmax)로 early / mid / late 세 개의 feature map을 생성. 어느 블록이 중요한지 모델이 스스로 학습 — 수렴 후 `get_block_importance()`로 시각화 가능.

**④ FPN Decoder**

늦게 뽑은 feature(semantic)를 먼저 처리하고, 위로 올라가면서 일찍 뽑은 feature(detail)를 합쳐 해상도를 복원. UNet의 skip connection과 같은 원리.

```
f_late (32×32)
  → ×2 upsample → cat(f_mid) → 64×64
  → ×2 upsample → cat(f_early) → 128×128
  → ×2 upsample → 256×256
  → ×2 upsample → 512×512
```

**⑤ 두 개의 mask 출력**

같은 FPN 구조 decoder를 독립적으로 두 개 사용.

- `stroke_mask` — 선이 어디 있는가
- `matte_pred`  — 어디가 머리카락인가

**⑥ color sampling → sketch_pred**

```
hair × matte_pred  →  머리카락 영역만 남김
→ 16×16 avg pool → nearest upsample → color_map
→ color_map × stroke_mask → sketch_pred
```

선이 있는 위치에, 그 위치 주변 머리카락의 패치 평균 색을 붙임. 학습 파라미터 없음.

### LoRA (OOD 문제 해결)

DiT는 denoiser로 학습됨 — `sigma=0` clean latent는 out-of-distribution.  
**24블록 전체**의 `to_q / to_k / to_v`에 rank-8 LoRA를 주입. 모든 블록이 동일한 OOD 입력을 처리하므로 3블록만 적응시키는 것은 일관성이 없음.  
대안(`sigma=ε` 미세 노이즈 주입)은 [한계 섹션](#62-lora와-ood) 참고.

---

## 3. GT 데이터 파이프라인

| 필드 | 생성 방법 |
|---|---|
| `sketch_GT` | 작가가 직접 그린 컬러 stroke 이미지 (PNG)|
| `matte_GT`  | hair segmentation mask — hair parser (BiSeNet 계열) 또는 수작업 |
| `stroke_mask_GT` | `sketch_GT`의 non-background 픽셀 자동 추출: `max(RGB) > 0.05` |

sketch_GT, matte_GT 품질은 검증 완료. stroke_mask_GT는 sketch_GT에서 자동 추출.

---

## 4. 학습 구성

### Phase 1 (epoch 0 ~ cycle_start): 직접 지도

| Loss | 가중치 | 내용 |
|---|---|---|
| BCE (structure) | 1.0 | stroke_mask vs GT |
| L1 (color) | 0.5 | sketch_pred vs sketch_GT, stroke 영역만 |
| BCE + L1 + Dice (matte) | 1.0 | matte_pred vs matte_GT |
| TV | 0.01 | stroke mask 연속성 (선 끊김 방지) |

> **주의**: matte loss는 BCE + L1 + Dice 세 항의 합에 w=1.0이 곱해짐. 실질적으로 structure loss(단일 항 × 1.0)의 약 3배 가중. 의도적 설계인지 검토 필요. 의도가 아니라면 각 항에 개별 가중치를 두거나 w_matte를 낮춰야 함.

### Phase 2 (epoch cycle_start ~): + Cycle Loss

```
sketch_pred → frozen forward ControlNet → block_features_pred
sketch_GT   → frozen forward ControlNet → block_features_GT
cycle_loss  = MSE(block_features_pred, block_features_GT)   w=0.01
```

forward ControlNet이 "올바르게 읽을 수 있는 sketch"를 inverse가 생성하도록 유도.  
**전제 조건**: forward ControlNet이 수렴된 상태여야 cycle signal이 유효 (→ [한계 섹션](#64-cycle-loss-설계의-전제-조건))

### 실행

```bash
# 선행 조건: checkpoints/phase2_braid/final.pth 존재
accelerate launch scripts/train_inverse_head.py --config configs/inverse_head.yaml
```

---

## 5. 실험 계획

### 우선순위 순서

| 순위 | 실험 | 이유 |
|---|---|---|
| 0 | **GT 품질 검증** | 학습 전 필수. GT가 나쁘면 모든 실험 무효 |
| 1 | **LoRA vs sigma=ε** | 구현 비용 최소, 핵심 설계 결정에 직접 영향 |
| 2 | **cycle_start 민감도** | 학습 중 병렬 실행 가능 (epoch 10 / 20 / 30) |
| 3 | **Color sampling grid** | 결과 보고 결정해도 늦지 않음 (8×8 / 16×16 / 32×32) |
| 4 | **Hook 위치 조합** | 계산 비용 높음, 마지막 |

### 핵심 지표

| 지표 | 측정 대상 | 방향 |
|---|---|---|
| Stroke F1 | stroke_mask_pred vs GT | ↑ |
| SSIM | sketch_pred vs sketch_GT | ↑ |
| LPIPS (sketch) | sketch_pred vs sketch_GT | ↓ |
| **LPIPS (round-trip)** | hair_recon vs hair_original | ↓ ← 핵심 |

### Round-trip 평가

```
hair → inverse → sketch_pred → forward → hair_recon
→ LPIPS(hair_recon, hair_original)
```

---

## 6. 알려진 한계 및 미검증 가정

### 6.1 Non-parametric Color Sampling

16×16 avg pool은 공간 해상도가 32px/patch로 극히 낮다.

- 배경 픽셀 혼입 가능성: 머리카락 경계 근처에서 배경 색이 섞임
- strand 수준의 색 변화(하이라이트, 그라데이션) 표현 불가

**선택 이유**: 학습 데이터 부족 상황에서 color 과적합 방지. 데이터가 충분하면 경량 color decoder로 대체 검토.

### 6.2 LoRA와 OOD

LoRA rank-8이 OOD 문제를 해결한다는 것은 가설. 검증 없이 채택.

**미검토 대안**: `sigma=ε` (예: 0.01) 미세 노이즈를 latent에 더해 DiT distribution 내부로 이동. 구현 단순하고 LoRA 파라미터 불필요 — 더 단순한 baseline.
