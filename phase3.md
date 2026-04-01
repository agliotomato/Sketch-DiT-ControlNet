# Phase 3: Compositing & 최종 평가

## Phase 2 체크포인트로 무엇을 하고 있는가

### Phase 2 산출물

| 산출물 | 경로 | 설명 |
|---|---|---|
| ControlNet 체크포인트 | `checkpoints/dit/phase2_braid/final.pth` | braid fine-tuned HairControlNet 가중치 |
| Hair Patch | `custom_results/dit/{stem}_gen.png` | sketch + matte 조건부 생성된 512×512 hair region |

### 전체 파이프라인 (Phase 3 포함)

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
- 별도의 background encoder 브랜치 존재

**우리 모델 (ControlNet residual injection)**:
```
ctrl_cond = sketch_latent + matte_feat
HairControlNet(noisy_latent, ctrl_cond) → residuals[0..11]
SD3.5 Transformer: hidden_states += residuals[int(i / 2)]  (24블록 전체)
```
- sketch를 frozen VAE로 latent 압축 → structural condition
- matte를 MatteCNN(3-layer conv)으로 16ch latent feature 변환 → spatial masking
- ControlNet residual을 transformer 24블록 전체에 2블록 간격으로 주입

### 3. Matte 활용 방식

| 항목 | 기존 논문 | 우리 모델 |
|---|---|---|
| matte 역할 | feature blending 가중치 (네트워크 내부) | spatial conditioning signal (MatteCNN) |
| 배경 처리 | Gaussian noise로 대체한 BG 입력 | 없음 (hair patch만 생성 후 후처리 합성) |
| 합성 위치 | 네트워크 내부 (feature level) | 외부 후처리 (image level + boundary inpainting) |

### 4. 색 제어 (StrokeColorSampler)

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
