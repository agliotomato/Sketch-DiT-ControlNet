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

## Component Details

### 1. SD3.5 VAE (Frozen)

- **Model**: `stabilityai/stable-diffusion-3.5-medium`, subfolder `vae`
- **Latent space**: 16ch
- **Spatial compression**: 8x (512x512 → 64x64)
- **Encoding formula**:
  ```
  latent = (raw_latent - shift_factor) * scaling_factor
  shift_factor   = 0.0609
  scaling_factor = 1.5305
  ```
- **역할 (이중)**:
  1. target 이미지 → latent 압축 (학습 supervision)
  2. sketch → latent 압축 (ControlNet conditioning input)

---

### 2. MatteCNN (Trainable)

```
Input:  matte (B, 1, 512, 512)   [0, 1]

Conv2d(1→16,  k=3, s=2, p=1) + GroupNorm(4)  + SiLU   # 512→256
Conv2d(16→32, k=3, s=2, p=1) + GroupNorm(8)  + SiLU   # 256→128
Conv2d(32→16, k=3, s=2, p=1) + GroupNorm(4)  + SiLU   # 128→64

Output: matte_feat (B, 16, 64, 64)
```

- **파라미터 수**: ~100K
- **역할**: "어느 공간 영역에 머리카락을 생성할 것인가"를 16ch latent 공간 신호로 변환
- sketch_latent와 add 후 ctrl_cond로 사용

---

### 3. HairControlNet (Trainable)

**초기화**: `SD3ControlNetModel.from_transformer(transformer, num_layers=12)`
- SD3.5-medium transformer의 첫 12개 블록 가중치를 복사
- 각 블록의 output projection은 **0으로 초기화** (zero-init)
  - 학습 초기: residual ≈ 0 → transformer에 영향 없음
  - 학습하면서 점진적으로 sketch 방향으로 조향

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

**Null text conditioning**:
- 텍스트 조건 없음 → learned null embedding 사용
- `null_encoder_hidden_states`: `nn.Parameter (1, 333, 4096)` — 학습됨
- `null_pooled_projections`: `nn.Parameter (1, 2048)` — 학습됨

---

## Training: Phase 1 (Unbraid Pretrain)

### Goal

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
  파라미터 업데이트: ❌
  하지만 residual 주입 연산을 통해 gradient는 통과:
    hidden_states += residuals[int(i / 2)]  ← 이 덧셈을 통해 역전파
  ↓
HairControlNet (12 blocks) → ✅ 파라미터 업데이트
  ↓
ctrl_cond = sketch_latent + matte_feat
  ├── sketch_latent: frozen VAE 출력 → ❌ (gradient 없음)
  └── matte_feat: MatteCNN 출력 → ✅
       ↓
MatteCNN → ✅ 파라미터 업데이트

null_encoder_hidden_states (nn.Parameter) → ✅
null_pooled_projections    (nn.Parameter) → ✅
```

**핵심**: PyTorch는 frozen 모듈을 통과할 때 모듈 **파라미터**의 gradient는 계산하지 않지만,
텐서 연산의 computation graph는 유지한다.
따라서 `hidden_states += residuals[i]` 덧셈을 통해 gradient가 residuals까지 역전파되고,
결과적으로 HairControlNet 파라미터가 업데이트된다.
| null_encoder_hidden_states | 333 x 4096 | **Trainable** |
| null_pooled_projections | 2048 | **Trainable** |

### Noise Schedule: Flow Matching

SD3.5는 DDPM이 아닌 **Rectified Flow (Flow Matching)** 사용.

**Forward process (noise 추가)**:
```
x_t = (1 - sigma) * x_0 + sigma * noise

sigma in [0, 1]:  sigma=0 → clean,  sigma=1 → pure noise
```

**Model prediction target (velocity)**:
```
v_target = noise - x_0
```

**x_0 복원** (LPIPS loss 계산용):
```
x_0_pred = x_t - sigma * v_pred
```

**Timestep sampling**: Logit-Normal distribution
```python
u = sigmoid(Normal(mean=0.0, std=1.0))   # [0, 1]
sigma = scheduler.sigmas[int(u * T)]
```
uniform보다 중간 sigma 구간에 샘플 집중 → 구조 학습에 유리.

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

### Loss Function

```
L_total = L_flow + w_lpips * L_lpips

L_flow (항상 활성):
  diff   = (v_pred - v_target)^2              # (B, 16, 64, 64)
  weight = matte_latent * 1.0
         + (1 - matte_latent) * 0.1           # 외부 영역 gradient 억제
  L_flow = mean(weight * diff)

L_lpips (steps의 30% 이후 활성):
  x0_pred  = x_t - sigma * v_pred             # flow matching x0 복원
  pred_rgb = VAE.decode(x0_pred)              # (B, 3, 512, 512)
  L_lpips  = LPIPS_VGG(pred_rgb * matte, target * matte)

w_lpips = 0.1
L_edge  = 0.0  (Phase 1 비활성)
```

**matte weighting 이유**: 배경(검정)은 trivially easy → gradient가 머리 영역에 집중되어야 함.

**LPIPS warmup 이유**: 초기에는 velocity 학습을 먼저 안정화, 이후 pixel-level 품질 추가.

### Optimizer & Schedule

```
Optimizer:  AdamW (beta1=0.9, beta2=0.999, weight_decay=1e-2)
LR:         1e-4, cosine decay to 1e-6
Warmup:     500 steps (linear)
Mixed prec: bf16
Grad accum: 2 steps
Grad clip:  1.0
EMA decay:  0.9999 (HairControlNet에만 적용)
Epochs:     200
Batch size: 4
```

### Data Augmentation (Phase 1)

| Augmentation | 확률 | 역할 |
|---|---|---|
| SketchColorJitter | p=0.8 | 색에 과적합 방지 — color = structural cue로 학습 |
| ThicknessJitter (dilation) | p=0.5 | 선 두께 변화에 robust |
| MattePerturbation (elastic) | p=0.3 | 경계 흔들림에 robust |
| AppearanceJitter | p=0.5 | 구조-외관 분리 (target에만 적용) |

Phase 2에서는 braid topology 보존을 위해 각 확률 낮춤.

### Evaluation Metrics (Phase 1)

| Metric | 측정 대상 |
|---|---|
| **SHR** (Sketch Hit Rate) | 스케치 선 위치에 생성 이미지의 edge가 존재하는가 |
| **MCS** (Matte Coverage Score) | 생성된 머리가 matte 영역 안에 있는가 |
| **LPIPS** | matte 내부 시각적 품질 |

---

## Training: Phase 2 (Braid Finetune)

Phase 1 완료 후 braid 데이터 (1K)로 fine-tune.

- Phase 1 checkpoint 로드 (`checkpoints/phase1_unbraid/best.pth`)
- LR 5x 낮춤 (2e-5) → unbraid prior 보존
- Edge loss 활성화 (w=0.05) → braid 구조 fidelity 강화
- Augmentation 강도 낮춤 → braid topology correspondence 보존
- 추가 metric: **BSS** (Braid Structure Score) — strand crossing 재현도

---

## File Structure

```
src/
  models/
    vae_wrapper.py        # SD3.5 VAE (16ch, frozen)
    controlnet_sd35.py    # HairControlNet + MatteCNN
  data/
    dataset.py            # HairRegionDataset (sketch, matte, target triplets)
    augmentation.py       # 4종 augmentation pipeline
    utils.py              # soft_composite, resize_matte_to_latent
  training/
    trainer.py            # Unified trainer (Phase 1 & 2)
    losses.py             # FlowMatchingLoss + PerceptualLoss + SketchEdgeAlignmentLoss
    ema.py                # EMA for HairControlNet

configs/
  base.yaml               # 공통 설정 (model_id, mixed_precision 등)
  phase1_unbraid.yaml     # Phase 1 config
  phase2_braid.yaml       # Phase 2 config

scripts/
  train.py                # python scripts/train.py --config configs/phase1_unbraid.yaml
  evaluate.py             # SHR, MCS, LPIPS, BSS
  infer.py                # single sample inference
  composite.py            # hair region → face image 합성
  check_dataset.py        # dataset triplet 검증
```
 