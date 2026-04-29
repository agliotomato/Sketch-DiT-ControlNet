# Hair-DiT Meeting 내용 정리

**Date:** 2026-04-07  
---

## 1. 학습 전략 및 Loss Function 현황
현재 모델은 헤어 영역에 집중된 **Masked Reconstruction** 방식을 채택하고 있습니다.

| 항목 | 상세 내용 | 비고 |
| :--- | :--- | :--- |
| **Target** | Ground Truth 이미지에서 Hair Matte 영역만 추출하여 학습 | |
| **MSE Loss** | Pixel-wise L2 Distance (Masked 영역 내) | Likelihood 최적화 |
| **LPIPS Loss** | 전체 이미지에 대해 적용 | 경계면 피처 왜곡 방지 목적 |
| **Edge Loss** | Sketch와의 정렬(Alignment) 강화 | |

---

## 2.Latent Space Composition
이미지 도메인의 매트와 레이턴트 도메인($64 \times 64$) 간의 해상도 차이로 인해 경계면에서의 정교한 합성이 필요합니다.

> [!IMPORTANT]
> **가설:** 단순 합성 시 경계면 피처가 튀거나 부자연스러울 수 있음.  
> **해결 방안:** **Weighted Sum Interpolation** 적용.

- 각 레이턴트 픽셀이 포함하는 매트 영역의 실제 비율(Area Percentage)을 계산하여 가중치 할당.
- ControlNet에서 추출된 24개 블록의 Residuals(Hidden States) 주입 시 해당 가중치를 반영하여 배경과의 조화 유도.

---

## 3. 주요 검증 과제 및 실험 결과 (Status Table)

아래 표는 주요 변수에 따른 실험 결과를 기록하기 위한 테이블입니다.

| Exp ID | Configuration | Sketch Alignment (0-1) | Boundary Naturalness | Texture Detail | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | Standard Reconstruction | | | | [Empty] |
| **WS-I** | Weighted Sum Interpolation | | | | [Empty] |
| **WS-E** | WS-I + Edge Loss | | | | [Empty] |
| **Full** | Hybrid Loss (MSE+LPIPS+Edge) | | | | [Empty] |

---

## 4. Context Dependency 문제 분석

### 현재 구조의 본질적 이슈 제기

Hair-DiT2의 inpainting 수식은 다음과 같습니다.

$$\text{noisy\_latents} = \underbrace{\text{latents} \cdot (1 - m)}_{\text{배경/인물 원본 유지}} + \underbrace{\sigma \cdot \epsilon \cdot m}_{\text{hair 영역만 noise 주입}}$$

여기서 $m$은 matte latent, $\sigma$는 noise schedule, $\epsilon \sim \mathcal{N}(0, I)$ 입니다.

이 구조에서 파생되는 핵심 문제는 다음과 같습니다.

| 문제 | 설명 |
| :--- | :--- |
| **Context Conditioning** | Hair 생성이 고정된 배경/인물 context에 암묵적으로 conditioned됨 |
| **입력 민감성** | 동일한 hair sketch 조건이라도 배경/인물 context가 달라지면 생성 결과가 달라질 수 있음 |
| **Global Attention 간섭** | DiT의 full self-attention은 hair 영역 토큰이 배경 토큰 전체를 참조하므로, 배경 정보가 hair 생성 경로에 직접 개입 |

> [!NOTE]
> 이는 구조적 설계상 불가피한 특성이나, **sketch 제어력이 context에 얼마나 종속되는지** 정량적으로 검증되지 않은 상태입니다.

---

## 5. 실험 설계: Context Dependency 검증

### Hypothesis 1: Face Context Dependency

**가설:** 동일한 hair sketch를 입력으로 주더라도, face context(표정, 포즈 등)가 달라지면 생성되는 hair가 달라진다.

```python
# 실험 설계
same_sketch = load_sketch("braid_001.png")
different_faces = [
    "face_front.jpg",    # 정면
    "face_side.jpg",     # 측면
    "face_smile.jpg",    # 웃는 얼굴
    "face_neutral.jpg"   # 무표정
]

# 예상 결과: 동일 sketch → 다른 hair 생성
```

**검증 지표:**

| 지표 | 설명 | 기대 방향 |
| :--- | :--- | :--- |
| **Sketch Alignment Score** | 생성 hair와 input sketch 간 edge 일치율 | face 변화에도 일정해야 함 |
| **Hair Shape Variance** | face 조건 간 생성 hair shape의 분산 | 낮을수록 sketch 제어력이 강함 |
| **DINO Feature Distance** | face context 간 생성 결과의 feature 거리 | context dependency 정도를 정량화 |

> [!IMPORTANT]
> 이 실험의 목적은 문제를 드러내는 것이 아니라, **sketch 제어력이 face context에 얼마나 robust한지** 수치로 확인하는 것입니다. 결과에 따라 추가적인 conditioning 전략(attention masking, region-specific guidance 등)을 검토합니다.

---

## 6. 실험: Baseline 대비 우위 점수 증명
- **비교 대상:** 현재 사용 가능한 헤어 생성 레퍼런스 모델.
- **평정 지표:** 스케치 일치도, 배경과의 경계 자연스러움, 생성된 헤어의 디테일(Texture).

---
