# ControlNet 역방향 학습: 헤어 사진 → 스케치

---

## 1. 연구 동기 및 문제 정의

스케치 기반 헤어 합성은 디퓨전 기반 ControlNet 구조를 통해 상당한 성과를 거뒀다. 그러나 근본적인 비대칭성이 남아 있다. **스케치 → 헤어 생성은 해결됐지만, 역방향인 실제 헤어 사진 → 구조화된 스케치 추출은 미해결 문제다.**

이 비대칭성은 실제 사용성에 결정적인 장벽을 만든다. 스케치 기반 합성 시스템을 쓰려면 사용자가 직접 색이 입혀진 선 스케치를 그려야 하는데, 이는 대부분의 사용자에게 없는 예술적 역량을 요구한다. 자동 역방향 모델이 있다면 다음이 가능해진다:

1. **무(無)노력 스타일 편집**: 참고 사진 업로드 → 편집 가능한 스케치 자동 생성 → 수정 후 재합성
2. **데이터 증강**: 레이블 없는 헤어 이미지에 스케치 어노테이션 자동 생성, 수작업 없이 학습 데이터 확장
3. **왕복 일관성 평가**: 스케치 기반 생성 시스템의 인코딩 충실도를 측정하는 정량 지표

단순하게는 기존 paired 데이터를 활용해 별도의 이미지-스케치 변환기(pix2pix 등)를 학습할 수 있다. 그러나 이 접근법은 근본적으로 불완전하다. **이미 학습된 forward 모델에 인코딩된 생성적 사전 지식을 완전히 무시하기 때문이다.** 시각적으로 GT 스케치와 비슷한 스케치가 반드시 최선의 헤어 생성을 이끌어내지는 않는다. 중요한 것은 예측된 스케치가 ControlNet 내부에서 GT 스케치와 동일한 *conditioning 신호*를 유발하는가이다.

이것이 **ControlNet Conditioning Space Inversion** 프레임워크의 동기다.

---

## 2. 문제 공식화

$\mathcal{F}: (\mathbf{s}, \mathbf{m}) \rightarrow \mathbf{x}$를 forward Sketch-to-Hair 모델로 정의한다. 여기서 $\mathbf{s} \in \mathbb{R}^{3 \times H \times W}$는 색이 입혀진 스케치, $\mathbf{m} \in \mathbb{R}^{1 \times H \times W}$는 헤어 마테, $\mathbf{x} \in \mathbb{R}^{3 \times H \times W}$는 합성된 헤어 이미지다.

우리는 **역방향 어댑터** $\mathcal{G}_\theta: \mathbf{x} \rightarrow (\hat{\mathbf{s}}, \hat{\mathbf{m}})$를 학습하여:

$$\mathcal{F}(\hat{\mathbf{s}}, \hat{\mathbf{m}}) \approx \mathbf{x}$$

가 성립하도록 한다. 핵심은 $\mathcal{G}_\theta$가 픽셀 공간만이 아니라 **ControlNet의 conditioning 공간**에서 작동해야 한다는 것이다. Feature 추출 함수를 다음과 같이 정의한다:

$$\Phi(\mathbf{s}, \mathbf{m}) = \{\mathbf{b}_i\}_{i=1}^{12}, \quad \mathbf{b}_i \in \mathbb{R}^{B \times 16 \times 64 \times 64}$$

여기서 $\{\mathbf{b}_i\}$는 frozen HairControlNet이 $(\mathbf{s}, \mathbf{m})$를 입력받아 생성하는 12개의 잔차 블록 출력(block samples)이다. 이 block samples가 frozen SD3.5 트랜스포머에 실제로 주입되는 conditioning 신호다.

역방향 목표는 두 가지 상호 보완적 요소로 구성된다:
- **픽셀 공간 충실도**: $\hat{\mathbf{s}}$가 시각적으로 GT 스케치와 유사해야 함
- **Feature 공간 일관성**: $\Phi(\hat{\mathbf{s}}, \hat{\mathbf{m}})$이 GT 쌍 $(\mathbf{s}^*, \mathbf{m}^*)$의 $\Phi(\mathbf{s}^*, \mathbf{m}^*)$와 일치해야 함

---

## 3. 방법론

### 3.1 아키텍처 개요

HairInversionAdapter $\mathcal{G}_\theta$는 frozen VAE 인코더 위에 구축된 세 개의 경량 학습 모듈로 구성된다.

```
헤어 이미지 x ∈ R^{3×512×512}
      │
      ▼  frozen VAE 인코더 (SD3.5-medium, 스케일링: (z - 0.0609) × 1.5305)
hair_latent z ∈ R^{16×64×64}
      │
      ▼  AdapterEncoder E_θ  (~15M 파라미터)
features f ∈ R^{256×64×64}
      │
      ├──▶  StrokeStructureDecoder D^s_θ  →  stroke_mask_raw ∈ R^{1×512×512}
      │         ↓ matte 게이팅: constrained_mask = stroke_mask_raw × matte_pred
      │         ↓ 비모수적 패치 색 샘플링 C(x, constrained_mask)
      │     sketch_pred ŝ ∈ R^{3×512×512}
      │
      └──▶  MatteDecoder D^m_θ           →  matte_pred m̂ ∈ R^{1×512×512}
```

**AdapterEncoder $E_\theta$**: VAE의 16채널 latent를 64×64 해상도의 256채널 feature map으로 변환한다. 공간 해상도를 유지(다운샘플링 없음)하며, 프로젝션 컨볼루션과 세 개의 잔차 블록으로 채널을 단계적으로 확장한다 (16→64→128→256).

**StrokeStructureDecoder $D^s_\theta$**: 어디에 선을 그을지를 예측한다. 세 개의 컨볼루션 레이어로 256→64→32→1 채널로 축소하고, sigmoid 활성화 및 8배 bilinear 업샘플링으로 512×512 해상도를 복원한다.

**MatteDecoder $D^m_\theta$**: $D^s_\theta$와 대칭적 구조. 소프트 헤어 경계 마스크를 예측한다.

### 3.2 비모수적 색 샘플링 (+ Matte 게이팅)

구조 예측(학습 가능)과 색 부여(비모수적)를 분리하는 것이 핵심 설계 결정이다. 이 결정의 근거는 forward 모델 학습 시 StrokeColorSampler 증강이 다음 제약을 강제한다는 관찰에서 온다:

$$\text{stroke\_color}(r) \approx \text{mean}\{x_p : p \in r\}$$

여기서 $r$은 스트로크 영역이고 $x_p$는 위치 $p$의 헤어 이미지 픽셀이다. 이 매핑을 별도의 색 헤드로 학습하는 대신 (분포 불일치 위험), 다음과 같이 해석적으로 구현한다.

색 샘플링 직전, stroke_mask_raw를 matte_pred로 게이팅하여 constrained_mask를 만든다:

$$\hat{\mathbf{s}}_\text{constrained} = \hat{\mathbf{s}}_\text{stroke} \odot \hat{\mathbf{m}}$$

$$C(\mathbf{x}, \hat{\mathbf{s}}_\text{constrained}) = \text{NearestUpsample}_{H \times W}\!\left(\text{AvgPool}_{G \times G}(\mathbf{x})\right) \odot \hat{\mathbf{s}}_\text{constrained}$$

여기서 $\hat{\mathbf{s}}_\text{stroke}$는 StrokeStructureDecoder의 raw sigmoid 출력이고, $G = 16$은 패치 크기를 정의한다 (각 32×32 영역이 해당 위치의 평균 헤어 색을 받음).

**Matte 게이팅의 필요성**: 이 단계는 단순한 구현 디테일이 아니다. 게이팅 없이 raw stroke_mask를 쓰면 배경 영역($\hat{\mathbf{m}} \approx 0$)에도 선이 그어지고, 그 배경 색이 스케치에 칠해진다. Forward 모델은 학습 중 배경이 포함된 스케치를 한 번도 보지 않았으므로, 이는 분포 불일치를 유발한다. $\hat{\mathbf{s}}_\text{constrained} = \hat{\mathbf{s}}_\text{stroke} \odot \hat{\mathbf{m}}$는 두 출력 헤드를 **아키텍처적으로 강결합**시켜, stroke와 matte 중 하나가 오동작하면 color sampling 오차가 $\mathcal{L}_\text{color}$를 통해 둘 다에 즉시 전파되는 구조를 만든다.

이 연산은 $\hat{\mathbf{s}}_\text{constrained}$에 대해 미분 가능하여, 색 예측을 통해 StrokeStructureDecoder와 MatteDecoder **양쪽 모두**에 gradient가 흐른다.

**핵심 장점**: 추론 시, 각 예측 스트로크 영역의 색이 입력 사진의 실제 헤어 색 분포 안에 놓임이 보장되며, 스트로크는 반드시 마테가 예측하는 헤어 영역 내에만 존재한다. 이는 forward 모델 학습 시 보았던 분포와 정확히 일치한다.

### 3.3 2단계 손실 스케줄

학습은 두 단계로 진행된다.

**Phase A — 직접 지도학습** (epoch 0 ~ $T_B$):

$$\mathcal{L}_A = \lambda_s \mathcal{L}_\text{structure} + \lambda_c \mathcal{L}_\text{color} + \lambda_m \mathcal{L}_\text{matte}$$

$$\mathcal{L}_\text{structure} = \text{BCE}(\hat{\mathbf{s}}_\text{bin},\, \mathbf{s}^*_\text{bin})$$

$$\mathcal{L}_\text{color} = \frac{\|\,(\hat{\mathbf{s}} - \mathbf{s}^*) \odot \mathbf{s}^*_\text{bin}\,\|_1}{\|\mathbf{s}^*_\text{bin}\|_1 + \epsilon}$$

$$\mathcal{L}_\text{matte} = \text{BCE}(\hat{\mathbf{m}}, \mathbf{m}^*) + \|\hat{\mathbf{m}} - \mathbf{m}^*\|_1 + \mathcal{L}_\text{Dice}(\hat{\mathbf{m}}, \mathbf{m}^*)$$

$$\mathcal{L}_\text{Dice} = 1 - \frac{2\sum_p \hat{m}_p m^*_p + \epsilon}{\sum_p \hat{m}_p + \sum_p m^*_p + \epsilon}$$

여기서 $\mathbf{s}^*_\text{bin} = \mathbf{1}[\max_c \mathbf{s}^*_c > 0.05]$는 GT 스케치에서 도출한 이진 스트로크 마스크다.

**왜 Dice loss가 필요한가**: 입력을 전체 사진(`batch["img"]`)으로 쓰면 마테 학습의 성격이 segmentation으로 바뀐다. 배경(대다수) vs. 헤어(소수)의 극심한 클래스 불균형 하에서, BCE + L1만으로는 "모든 픽셀 = 0" 예측이 낮은 loss를 기록하는 trivial solution이 존재한다. Dice loss는 클래스 비율에 무관하게 예측-GT 간 겹침 비율을 직접 최적화하여 이를 방지한다. $\lambda_m = 2.0$으로 높게 설정하는 것도 같은 이유 — 마테가 무너지면 color sampling 전체가 오동작한다.

**Phase B — Feature Cycle Consistency** (epoch $T_B$ ~ $T$, warm-up 포함):

$$\mathcal{L}_B = \mathcal{L}_A + \omega(t) \cdot \mathcal{L}_\text{feature}$$

$$\mathcal{L}_\text{feature} = \frac{1}{12} \sum_{i=1}^{12} \|\Phi_i(\hat{\mathbf{s}}, \hat{\mathbf{m}}) - \Phi_i(\mathbf{s}^*, \mathbf{m}^*)\|_2^2$$

warm-up 스케줄 $\omega(t)$는 Phase B 진입 후 $T_\text{warmup}$ epoch에 걸쳐 0에서 $\lambda_f$까지 선형 증가한다:

$$\omega(t) = \lambda_f \cdot \min\!\left(1,\, \frac{t - T_B}{T_\text{warmup}}\right)$$

이 스케줄링이 결정적으로 중요하다. $\lambda_f$를 갑자기 적용하면, 모델이 시각적으로는 열악한 스케치를 만들면서 ControlNet feature만 억지로 맞추는 **적대적 최적화**에 빠질 수 있다. Warm-up은 feature 공간 정렬이 강제되기 전에 픽셀 공간 품질이 먼저 안정되도록 보장한다.

### 3.4 구현 세부 사항

| 항목 | 값 |
|------|-----|
| **Frozen 구성요소** | VAE 인코더, HairControlNet (Phase 2 braid 체크포인트) |
| **학습 파라미터** | AdapterEncoder + StrokeStructureDecoder + MatteDecoder ≈ 15M |
| **옵티마이저** | AdamW, lr = $10^{-4}$, weight decay = $10^{-2}$ |
| **LR 스케줄** | 선형 warm-up (200 스텝) → cosine annealing |
| **배치 크기** | 8, bf16 혼합 정밀도 |
| **학습 데이터** | unbraid 3K + braid 1K = 4K 쌍 (커리큘럼 불필요, 소형 지도학습 모델) |
| **손실 가중치** | $\lambda_s=1.0$, $\lambda_c=0.5$, $\lambda_m=2.0$ (Dice 포함), $\lambda_f=0.1$ |
| **Phase B 설정** | $T_B=10$, $T_\text{warmup}=10$ |

---

## 4. 기여 및 Novelty

**① ControlNet Conditioning Space Inversion**
기존 DDIM Inversion(Song et al., ICLR 2021)이 *노이즈 latent*를 역방향으로 푸는 것과 달리, 우리는 *conditioning 입력*을 역방향으로 학습한다. 특히 ControlNet의 12개 잔차 블록 출력 — 트랜스포머에 실제로 주입되는 공간 feature — 을 정렬 목표로 삼는 최초의 접근이다.

**② 학습 분포 인식 비모수적 색 샘플링**
Forward 학습 증강(StrokeColorSampler)을 추론 시점에서 해석적으로 재현함으로써, 예측 스케치의 색 분포가 forward 모델이 학습 시 본 분포와 정확히 일치함을 보장한다. 추가 파라미터 없이 분포 불일치를 원천 차단한다.

**③ 손실 Warm-up을 통한 점진적 Feature 정렬**
Conditioning 공간에서의 적대적 최적화(feature는 맞지만 시각적으로 열악) 실패 모드를 식별하고, 2단계 손실 스케줄 + warm-up으로 이를 방지한다. ControlNet inversion 문헌에서 이전에 다뤄지지 않은 실패 모드다.

**④ 왕복 평가 프로토콜 (Round-Trip Evaluation)**
역방향 어댑터가 새로운 평가 축을 가능하게 한다: *Round-Trip LPIPS* — $d(\mathcal{F}(\mathcal{G}(\mathbf{x})), \mathbf{x})$를 스케치 기반 생성 시스템의 자기 일관성 지표로 측정.

---

## 5. 선행 연구와의 비교

| 연구 | 역방향 대상 | Conditioning 유형 | Feature 공간 손실 |
|------|------------|------------------|------------------|
| DDIM Inversion [Song et al. 2021] | 노이즈 latent $z_T$ | 없음 (무조건) | ✗ |
| Null-text Inversion [Mokady et al. 2023] | 텍스트 임베딩 | CLIP 텍스트 | ✗ |
| IP-Adapter [Ye et al. 2023] | 이미지 프롬프트 임베딩 | CLIP 이미지 | ✗ |
| VisualBuddy / Vision Banana | Latent feature | 이미지 feature | 부분적 |
| **Ours** | **스케치 + 마테** | **ControlNet block samples** | **✓ (12-block MSE)** |

IP-Adapter와 가장 유사하다. 둘 다 생성 백본을 동결하고 이미지 공간 conditioning을 위한 경량 인코더를 학습한다. 차이점: (1) 우리는 semantic 이미지 임베딩이 아닌 *구조화된* 스케치+마테 conditioning을 역방향으로 학습; (2) feature 정렬 손실이 콤팩트 임베딩 벡터가 아니라 트랜스포머를 실제로 conditioning하는 *공간 잔차* 블록 출력에 작용.

---

## 6. 예상 결과 및 평가 계획

### 정량 지표
- **스케치 품질**: SSIM↑, Edge IoU↑ (GT 스케치 대비)
- **마테 품질**: IoU↑ (GT 마테 대비)
- **왕복 일관성**: $\mathcal{F}(\mathcal{G}(\mathbf{x}))$와 $\mathbf{x}$ 간 LPIPS↓
- **Ablation**: Phase A only vs Phase A+B(급격) vs Phase A+B(warm-up)

### 정성적 시연
```
실제 브레이드 사진 → [HairInversionAdapter] → 예측 (스케치, 마테)
                                                       ↓
                                      [HairControlNet + SD3.5]
                                                       ↓
                                          재합성된 헤어 사진
```
재합성된 헤어는 입력 사진의 브레이드 구조, 색 분포, 마테 경계를 보존해야 한다. 이것이 역방향 모델이 forward 모델의 conditioning 표현을 포착했음을 보인다.

---

## 7. 설계 교훈: 입력 변경은 loss 재설계를 수반한다

이번 inversion adapter 설계 과정에서 가장 중요한 교훈은 **입력 공간을 바꾸면 반드시 loss 공간도 함께 재설계해야 한다**는 것이다. 이 섹션은 실제 구현 과정에서 마주한 두 가지 설계 실수와 그 수정을 기록한다.

### 교훈 1: 입력 분포 변경 → 세그멘테이션 loss 필요

**초기 설계의 실수**: 마테 loss를 BCE + L1으로 설계하고 $\lambda_m = 1.0$으로 설정했다. 이는 입력이 `img × matte`(헤어 영역만 남긴 사진)라는 암묵적 가정에서 온 것이다.

**문제**: inversion의 입력은 `img`(전체 사진)이어야 한다. 실제 사용자는 헤어 마테를 미리 갖고 있지 않으며, inversion의 목적 자체가 마테를 예측하는 것이다. 전체 사진을 입력으로 쓰면:
- 배경 픽셀 ≫ 헤어 픽셀 (극단적 클래스 불균형)
- BCE + L1: "모든 픽셀 = 0 예측"이 낮은 loss를 기록하는 trivial solution 존재
- 모델이 배경 억제에 집중하다 헤어 경계 예측을 포기함

**수정**: Dice loss 추가 + $\lambda_m = 2.0$ 상향. Dice는 클래스 비율에 무관하게 겹침 비율을 직접 최적화 — 불균형 세그멘테이션의 표준 해법.

**일반화 원칙**: loss 함수는 입력의 분포를 암묵적으로 가정한다. 입력을 바꾸면 그 가정이 깨지므로, loss의 전제를 처음부터 다시 점검해야 한다.

### 교훈 2: 두 출력 헤드는 loss가 아닌 아키텍처로 강결합해야 한다

**초기 설계의 실수**: StrokeStructureDecoder와 MatteDecoder를 독립된 헤드로 두고, loss만으로 두 예측이 일관되기를 기대했다.

**문제**: 색 샘플링 시 raw stroke_mask를 직접 사용하면, StrokeStructureDecoder가 배경에 선을 그어도 MatteDecoder의 loss와 무관하게 된다. 배경 영역 스트로크는 $\mathcal{L}_\text{structure}$에서 패널티를 받지만, 실제로 배경 색이 스케치에 칠해지는 부작용은 downstream forward pass에서만 발현된다 — loss가 직접 포착하지 못하는 숨겨진 오차다.

**수정**: `constrained_mask = stroke_mask × matte_pred`를 색 샘플링의 입력으로 사용. 이로써:
- Gradient가 color sampling을 통해 **두 디코더 모두**에 흐름
- 배경 스트로크는 matte_pred = 0으로 즉시 억제됨 (별도 loss 불필요)
- 두 헤드의 오차가 서로의 학습에 즉각 전파되는 강결합 달성

**일반화 원칙**: 두 예측이 물리적으로 의존 관계에 있다면 (스트로크는 헤어 영역 안에만 있어야 한다), loss보다 **아키텍처에서** 그 제약을 구현하는 것이 더 강력하다. Loss는 확률적이고 배치 평균이지만, 아키텍처적 제약은 모든 픽셀, 모든 스텝에서 항상 적용된다.

### 요약

| 설계 결정 | 나쁜 버전 | 수정된 버전 | 핵심 이유 |
|-----------|----------|------------|---------|
| 마테 loss | BCE + L1, λ=1.0 | BCE + L1 + Dice, λ=2.0 | 전체 사진 입력 → 클래스 불균형 |
| 색 샘플링 | `stroke_mask` 직접 사용 | `stroke_mask × matte_pred` | 아키텍처가 물리적 제약 강제 |
| 입력 선택 | `img × matte` (헤어 영역) | `img` (전체 사진) | 실제 추론 시나리오와 일치 |

이 두 수정은 각각 독립적으로도 중요하지만, 함께 작동할 때 시너지가 있다: 마테 예측이 정확해야 → 색 샘플링 제약이 정확하게 적용되고 → 정확한 색 샘플링이 → $\mathcal{L}_\text{color}$를 통해 마테 예측 학습에 다시 도움이 된다.
