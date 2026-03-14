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
