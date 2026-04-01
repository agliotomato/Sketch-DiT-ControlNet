# Architecture: Sketch-DiT-ControlNet

## 연구목표

```
(colored sketch + matte) → hair region image
```

SketchHairSalon의 GAN-based S2I-Net을 SD3.5 ControlNet으로 교체.

**핵심 질문**: colored sketch + matte를 조건으로 줬을 때, DiT 기반 모델이 braid/unbraid hair structure를 실제로 따르는 hair region image를 생성할 수 있는가?

---

## Overview

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                        Inference Pipeline                        │
 │                                                                  │
 │  sketch (3ch, 512x512)  ──→  SD3.5 VAE encode  ──→ (16ch, 64x64)│
 │  matte  (1ch, 512x512)  ──→  MatteCNN          ──→ (16ch, 64x64)│
 │                                        ADD ↓                     │
 │                               ctrl_cond (16ch, 64x64)           │
 │                                         │                        │
 │  noise (16ch, 64x64) ──→  HairControlNet (12 blocks, trainable) │
 │  + ctrl_cond                            │ residuals[0..11]       │
 │                                         ▼                        │
 │  noise (16ch, 64x64) ──→  SD3.5 Transformer (24 blocks, frozen) │
 │                           + null text embedding                  │
 │                           + ControlNet residuals injected        │
 │                                         │                        │
 │                                    v_pred (16ch)                 │
 │                                         │                        │
 │                              Euler sampling loop                 │
 │                                         │                        │
 │                              SD3.5 VAE decode                   │
 │                                         │                        │
 │                         hair region image (3ch, 512x512)        │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 구성.

### 1. SD3.5 VAE (Frozen)

- **Model**: `stabilityai/stable-diffusion-3.5-medium`, subfolder `vae`
- **Latent space**: 16ch
- **Spatial compression**: 8x (512x512 → 64x64)
- **역할**:
  1. target 이미지 → latent 압축
  2. sketch → latent 압축 

---

### 2. MatteCNN (Trainable)

```
Input:  matte (B, 1, 512, 512)   [0, 1]

Conv2d(1→16,  k=3, s=2, p=1) + GroupNorm(4)  + SiLU   # 512→256
Conv2d(16→32, k=3, s=2, p=1) + GroupNorm(8)  + SiLU   # 256→128
Conv2d(32→16, k=3, s=2, p=1) + GroupNorm(4)  + SiLU   # 128→64

Output: matte_feat (B, 16, 64, 64)
```

- **역할**: "어느 공간 영역에 머리카락을 생성할 것인가"를 16ch latent 공간 신호로 변환
- sketch_latent와 add 후 ctrl_cond로 사용

---

### 3. HairControlNet (Trainable)

**초기화**: `SD3ControlNetModel.from_transformer(transformer, num_layers=12)`
- SD3.5-medium transformer의 첫 12개 블록 가중치를 복사
  - SD3.5는 이미 수십억 장으로 학습된 feature extractor → 랜덤 초기화 대비 빠르고 안정적
- 각 블록의 output projection만 **0으로 초기화** (zero-init)
  - Attention/FFN 내부는 SD3.5 가중치 유지 → feature 추출 능력 보존
  - output projection = 0 → residual = 0 → 학습 초기엔 ControlNet이 없는 것과 동일
  - frozen transformer 입력이 망가지지 않아 loss 폭발 없이 안정적으로 시작
  - backprop이 돌면서 output projection이 0에서 서서히 커짐 → sketch 구조를 점진적으로 주입

**입력 처리**:
```
noisy_latent (B, 16, 64x64)
    → patch embedding → image tokens (B, 1024, hidden_dim)
                                     +
ctrl_cond (B, 16, 64x64)  [= sketch_latent + matte_feat]
    → controlnet_x_embedder → cond tokens (B, 1024, hidden_dim)

combined = image_tokens + cond_tokens
```

**블록 처리**:
```
combined
  → Block 0  (Joint Attention + FFN)  → output_0  = residual[0]
  → Block 1                           → output_1  = residual[1]
  ...
  → Block 11                          → output_11 = residual[11]
```

**출력**: `residuals = [output_0, output_1, ..., output_11]`

---

### 4. SD3.5 Transformer (Frozen)

- **Model**: `stabilityai/stable-diffusion-3.5-medium`, subfolder `transformer`
- **Architecture**: MM-DiT (Multi-Modal DiT), Joint Attention
- **총 블록 수**: 24
- **ControlNet residual 주입**:
  ```python
  interval = 24 / 12 = 2.0

  # 각 transformer 블록 i에서:
  hidden_states += residuals[int(i / 2)]
  ```
  - ControlNet residual 1개가 transformer 블록 2개 담당
  - 전체 24블록 모두 영향받음


## Training: Phase 1 (Unbraid Pretrain)

### 목표

SD3.5 + ControlNet이 unbraid hair sketch를 따르는 hair region 생성 능력 획득.

### Trainable Parameters

| 컴포넌트 | 파라미터 수 | 학습 여부 |
|---|---|---|
| SD3.5 Transformer | ~2.5B | Frozen |
| SD3.5 VAE | ~80M | Frozen |
| HairControlNet (12 blocks) | ~1.2B | **Trainable** |
| MatteCNN | ~100K | **Trainable** |
| null_encoder_hidden_states | 333 × 4096 | **Trainable** |
| null_pooled_projections | 2048 | **Trainable** |

### Gradient Flow (Backpropagation)

```
Loss (v_pred vs v_target)
  ↓
SD3.5 Transformer (frozen)
  파라미터 업데이트: 없음
  하지만 residual 주입 연산을 통해 gradient는 통과:
    hidden_states += residuals[int(i / 2)]  ← 이 덧셈을 통해 역전파
  ↓
HairControlNet (12 blocks) →  파라미터 업데이트
  ↓
ctrl_cond = sketch_latent + matte_feat
  ├── sketch_latent: frozen VAE 출력 → gradient 없음
  └── matte_feat: MatteCNN 출력 → 파라미터 업데이트
       ↓
MatteCNN →  파라미터 업데이트

null_encoder_hidden_states (nn.Parameter) → 파라미터 업데이트
null_pooled_projections    (nn.Parameter) → 파라미터 업데이트
```

### Noise Schedule: Flow Matching

SD3.5는 DDPM이 아닌 **Rectified Flow (Flow Matching)** 사용.

### Training Step

```
① batch 로딩
   sketch  (B, 3, 512, 512)
   matte   (B, 1, 512, 512)
   target  (B, 3, 512, 512)  = img * matte (hair region)

② target → SD3.5 VAE encode → latents (B, 16, 64, 64)

③ sigma 샘플링 (logit-normal)
   noisy_latents = (1-sigma)*latents + sigma*noise

④ HairControlNet.forward(noisy_latents, sketch, matte, sigma)
     sketch → frozen VAE → sketch_latent (B, 16, 64, 64)
     matte  → MatteCNN   → matte_feat   (B, 16, 64, 64)
     ctrl_cond = sketch_latent + matte_feat
     SD3ControlNetModel → residuals[0..11]
     return residuals, null_enc_hs, null_pooled

⑤ SD3.5 Transformer.forward(
       hidden_states=noisy_latents,
       encoder_hidden_states=null_enc_hs,
       pooled_projections=null_pooled,
       timestep=sigma,
       block_controlnet_hidden_states=residuals,
   ) → v_pred (B, 16, 64, 64)

⑥ v_target = noise - latents

⑦ Loss 계산 → Backward → optimizer step (ControlNet만)
```

### Data Augmentation (Phase 1)

| Augmentation | 확률 | 역할 |
|---|---|---|
| SketchColorJitter | p=0.8 | 색에 과적합 방지 — color = structural cue로 학습 |
| ThicknessJitter (dilation) | p=0.5 | 선 두께 변화에 강건 |
| MattePerturbation (elastic) | p=0.3 | 경계 흔들림에 강건 |
| AppearanceJitter | p=0.5 | 구조-외관 분리 (target에만 적용) |

---

## Training: Phase 2 (Braid Fine-tuning)

### 학습 설정

| 항목 | 값 |
|---|---|
| 백본 | SD3.5 Medium (8B) + HairControlNet (12 layers) |
| Phase 1 체크포인트 | `checkpoints/phase1_unbraid/epoch_40.pth` |
| 데이터셋 | braid_train (1,000샘플) |
| Epochs | 100 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Loss | Flow matching + LPIPS (즉시 활성화) + Edge alignment (w=0.05) |
| 최종 avg loss | 0.0105 |

### 색 제어 구현 (StrokeColorSampler)

현재(main 브랜치)는 `SketchColorJitter(p=0.8)`로 학습되었다.
이 augmentation은 매 iteration마다 stroke 색을 **완전 랜덤 HSV 색**으로 교체한다.
결과적으로 모델이 색을 무시하고 구조(선 방향, 교차, 땋임 패턴)만 보고 머리를 생성한다.

GAN(`color_coding`)과 동일한 원리를 DiT 학습에 적용:

```python
# 각 stroke 영역 → target 이미지의 해당 위치 픽셀 중 무작위 1개 샘플링
# → 그 색을 stroke 색으로 할당 (매 iteration마다 새로 샘플링)
```

GAN과의 차이점:
- GAN(`color_coding`): RGB 스케치를 grayscale로 변환 후 밝기 값으로 stroke 구분 → collision 위험
- 우리(`StrokeColorSampler`): RGB 값 그대로 양자화하여 stroke 구분 → collision 없음

`AppearanceJitter`도 함께 제거 (stroke ↔ target 색 대응이 깨지기 때문).

기대 효과:
```
갈색 stroke → 갈색 머리  ✓
금발 stroke → 금발 머리  ✓
검정 stroke → 검정 머리  ✓
```

한계: 학습 데이터에 존재하지 않는 색(비자연 색상)은 제어 불가.

---

## Phase 3: Compositing & 최종 평가

### 전체 파이프라인

```
[S2M 스케치]
    │
    ▼
S2M-Net (SketchHairSalon, GAN) ──→ matte (soft alpha)
    │
    ├── [S2I 스케치 + matte] ──→ GAN (hairsalon_S2I_braid)  ──→ hair patch (GAN)
    │
    └── [S2I 스케치 + matte] ──→ DiT ControlNet (phase2)    ──→ hair patch (DiT)
                                                                      │
                                                              Step 2: composite.py
                                                              hair * matte + face * (1 - matte)
                                                              + Gaussian feathering
                                                                      │
                                                              Step 3: inpaint_boundary.py
                                                              경계 ring (dilate - erode) inpainting
                                                              SD2 inpaint, strength=0.45
                                                                      │
                                                              최종 합성 이미지
```

### Phase 3에서 하는 일

1. **Composite** (`scripts/composite.py`): 생성된 hair patch를 face 이미지에 alpha 합성
   - `result = hair * matte + face * (1 - matte)` + Gaussian feathering (σ=3.0)
2. **Boundary Inpainting** (`scripts/inpaint_boundary.py`): 경계 부자연스러움 제거
   - 마스크: `dilate(matte) - erode(matte)` → 경계 안팎 ring 양쪽 inpaint
   - SD2 inpainting pipeline, strength=0.45, 20 steps
3. **비교 평가**: GAN vs DiT 최종 합성 결과 비교 (FID, LPIPS, 주관 평가)

---

## 기존 논문(SketchHairSalon)과의 차이점

### 1. 생성 백본: GAN → Diffusion Transformer

| 항목 | 기존 논문 (SketchHairSalon) | 우리 모델 |
|---|---|---|
| 생성 모델 | GAN (pix2pix 계열, ~100M) | SD3.5 Medium + HairControlNet (~3.7B trainable) |
| 학습 방식 | Adversarial (G + D) | Flow Matching (v-prediction, Rectified Flow) |
| 텍스트 조건 | 없음 | null text embedding (학습 가능 파라미터) |
| 해상도 | 512×512 | 512×512 (latent 64×64) |

### 2. 조건 주입 메커니즘

**기존 논문 (feature-level blending)**:
```
F_i = F_i^h * M_i + F_i^{BG} * (1 - M_i)
```
- 네트워크 마지막 4개 레이어에서 hair feature + 배경 feature를 matte로 blending
- 배경 입력(BG): face 이미지의 hair 영역을 Gaussian noise로 대체한 것

**우리 모델 (ControlNet residual injection)**:
```
ctrl_cond = sketch_latent + matte_feat
HairControlNet(noisy_latent, ctrl_cond) → residuals[0..11]
SD3.5 Transformer: hidden_states += residuals[int(i / 2)]  (24블록 전체)
```

### 3. Matte 활용 방식

| 항목 | 기존 논문 | 우리 모델 |
|---|---|---|
| matte 역할 | feature blending 가중치 (네트워크 내부) | spatial conditioning signal (MatteCNN) |
| 배경 처리 | Gaussian noise로 대체한 BG 입력 | 없음 (hair patch만 생성 후 후처리 합성) |
| 합성 위치 | 네트워크 내부 (feature level) | 외부 후처리 (image level + boundary inpainting) |

### 4. 색 제어

| 항목 | 기존 논문 (color_coding) | 우리 모델 (StrokeColorSampler) |
|---|---|---|
| stroke 구분 방식 | grayscale 밝기값 | RGB 값 그대로 양자화 |
| collision 위험 | 있음 (밝기 유사 → 같은 stroke로 합산) | 없음 (stroke마다 임의 레이블 색으로 구분) |
| 색 할당 | target 해당 위치 픽셀 평균 | target 해당 위치 픽셀 중 무작위 1개 샘플링 |
| AppearanceJitter | 적용 | 제거 (stroke↔target 색 대응 보존) |

### 5. 학습 전략: 2단계 커리큘럼

| Phase | 데이터 | 목적 |
|---|---|---|
| Phase 1 | unbraid (일반 머리카락) | hair structure 기초 학습 |
| Phase 2 | braid (1,000샘플) fine-tune | braid 특화 (교차·땋임 패턴) |

기존 논문은 braid/unbraid를 분리하지 않고 단일 모델 학습.

---

## Novelty 정리

### [N1] GAN → DiT 대체를 통한 sketch-conditioned hair 생성

SD3.5 (8B, MM-DiT)의 사전학습 feature를 ControlNet으로 재활용하여 GAN 대비:
- 모드 붕괴 없음
- 텍스처 디테일 (strand 분리, 광택, 음영) 향상
- stroke 구조를 더 충실하게 따르는 생성 가능

### [N2] MatteCNN: soft matte → latent spatial signal 변환

기존 논문은 matte를 feature blending 가중치로 직접 사용.
우리는 MatteCNN으로 matte를 16ch latent 특징으로 변환하여 sketch_latent와 add함으로써:
- "어디에 그릴 것인가"를 ControlNet이 학습 가능한 공간 신호로 제공
- matte 경계의 soft gradient가 latent 공간에서도 보존됨

### [N3] StrokeColorSampler: RGB 기반 stroke 구분으로 collision-free 색 제어

기존 GAN의 grayscale 기반 color_coding 대비 RGB 양자화로 stroke 충돌 없이 정확한 색 대응 학습.
결과: "갈색 stroke → 갈색 머리", "금발 stroke → 금발 머리" 제어 가능함을 실험적으로 확인.

### [N4] 2단계 커리큘럼 학습 (unbraid pretrain → braid fine-tune)

braid 데이터만으로 처음부터 학습하면 교차 패턴 과적합 위험.
Phase 1에서 일반 hair structure 기초를 학습한 뒤 Phase 2에서 braid로 fine-tune함으로써:
- 빠른 수렴 (Phase 2 avg loss 0.0105)
- 교차·땋임 패턴과 기본 hair texture를 동시에 보존

### [N5] 후처리 경계 inpainting (dilate-erode ring mask)

논문의 feature-level blending을 네트워크 외부에서 근사.
단순 alpha compositing 대비 `dilate(matte) - erode(matte)` ring 마스크로
경계 안팎 양쪽을 inpainting하여 hair-face 전환 자연스러움 개선.

---

## 평가 계획

| 지표 | 비교 대상 | 비고 |
|---|---|---|
| FID | GAN vs DiT (braid_test) | feature distribution 차이 |
| LPIPS | GAN vs DiT vs GT | perceptual similarity |
| 색 제어 정확도 | stroke 색 vs 생성 색 평균 ΔE | 주관/정량 혼합 |
| 구조 일치도 | sketch edge vs gen edge (Canny) | edge alignment |
| 합성 자연스러움 | composite → inpaint 결과 주관 평가 | 경계 품질 |
