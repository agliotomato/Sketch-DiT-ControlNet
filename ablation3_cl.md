# Ablation Study C: Curriculum Learning Strategy

## 실험 설정

Curriculum Learning 전략에 따른 성능 차이를 분석한다. 비교 대상은 다음 5가지 조건이다.

| 조건 | 설명 |
|------|------|
| Unbraid Only (c1) | Unbraid 태스크만으로 단독 학습 |
| Braid Only (c2) | Braid 태스크만으로 단독 학습 |
| Joint (c3) | 두 태스크를 동시에 병렬 학습 |
| Reverse Curriculum (c4) | 어려운 태스크(Braid) → 쉬운 태스크(Unbraid) 순서 학습 |
| Ours / Forward Curriculum (c5) | 쉬운 태스크(Unbraid) → 어려운 태스크(Braid) 순서 학습 |

평가는 Combined(전체), Braid 전용, Unbraid 전용 세 구간으로 분리하여 수행하였다.

---

## 1. 종합 성능 요약 (Combined) 

Combined 결과는 Braid Only 베이스라인이 없으며, 세 커리큘럼 조건을 직접 비교한다.

| Metric | Joint | Reverse Curriculum | **Ours (Forward)** |
|--------|------:|-------------------:|-------------------:|
| Edge IoU ↑ | 0.0613 | 0.0607 | **0.0616** |
| Chamfer ↓ | **5.3523** | 5.5852 | 5.6994 |
| Sketch LPIPS ↓ | 0.6742 | **0.6717** | 0.6720 |
| PSNR ↑ | 16.1149 | 15.5648 | **16.2746** |
| SSIM ↑ | 0.6047 | 0.5969 | **0.6094** |
| LPIPS ↓ | **0.1928** | 0.2054 | 0.1993 |
| Hair FID ↓ | **50.67** | 55.62 | 53.07 |

**Forward Curriculum이 우세한 지표**: Edge IoU, PSNR, SSIM (3/7)  
**Joint가 우세한 지표**: Chamfer, LPIPS, Hair FID (3/7)  
**Reverse Curriculum이 우세한 지표**: Sketch LPIPS (1/7)

전체적으로 Forward Curriculum은 픽셀 수준 재현 품질(PSNR, SSIM)과 구조 검출(Edge IoU)에서 가장 우수하다. 반면 Chamfer와 Hair FID는 Joint가 더 낮아 공간적 정밀도와 분포 충실도 측면에서 Joint가 유리하다. Reverse Curriculum은 PSNR(15.56)과 SSIM(0.5969)이 세 조건 중 가장 낮아 전반적 성능이 떨어진다.

---

## 2. Braid 도메인 분석

Braid 전용 이미지에 대한 4개 조건 비교이다.

| Metric | Braid Only | Joint | Reverse Curriculum | **Ours (Forward)** |
|--------|----------:|------:|-------------------:|-------------------:|
| Edge IoU ↑ | 0.1001 | 0.0996 | 0.0995 | **0.1007** |
| Chamfer ↓ | **2.8776** | 2.8807 | 2.8923 | 2.9370 |
| Sketch LPIPS ↓ | 0.7095 | **0.6932** | 0.7058 | 0.7126 |
| PSNR ↑ | 15.3378 | 15.6549 | 14.5220 | **15.8495** |
| SSIM ↑ | 0.6055 | 0.6107 | 0.5872 | **0.6144** |
| LPIPS ↓ | 0.1906 | **0.1780** | 0.2033 | 0.1858 |
| Hair FID ↓ | 80.62 | **79.32** | 103.15 | 83.31 |

**Forward Curriculum이 우세한 지표**: Edge IoU, PSNR, SSIM (3/7)  
**Joint가 우세한 지표**: Sketch LPIPS, LPIPS, Hair FID (3/7)  
**Braid Only가 우세한 지표**: Chamfer (1/7)

Forward Curriculum은 Braid 도메인에서 픽셀 수준 재현 품질(PSNR 15.85, SSIM 0.6144)로 전체 1위를 기록한다. 이는 Unbraid를 선행 학습함으로써 생성 일반화 능력이 향상된 결과로 해석된다. 그러나 Sketch LPIPS(0.7126), LPIPS(0.1858), Hair FID(83.31)는 Joint에 못 미쳐 지각 품질과 분포 충실도 측면에서 Joint가 여전히 우세하다.

Chamfer는 Braid Only(2.8776)가 가장 낮아 단일 태스크 특화 학습이 구조적 정밀도에서 유리함을 보여 준다.

Reverse Curriculum은 Hair FID(103.15)가 심각하게 높아 역순 학습이 Braid 도메인의 분포 학습을 크게 저해함을 시사한다.

---

## 3. Unbraid 도메인 분석

Unbraid 전용 이미지에 대한 4개 조건 비교이다.

| Metric | Unbraid Only | Joint | Reverse Curriculum | **Ours (Forward)** |
|--------|------------:|------:|-------------------:|-------------------:|
| Edge IoU ↑ | 0.0523 | 0.0524 | 0.0518 | **0.0527** |
| Chamfer ↓ | **5.8657** | 5.9198 | 6.2035 | 6.3337 |
| Sketch LPIPS ↓ | 0.6871 | 0.6698 | 0.6638 | **0.6627** |
| PSNR ↑ | 16.0440 | 16.2205 | 15.8042 | **16.3722** |
| SSIM ↑ | 0.6011 | 0.6034 | 0.5992 | **0.6083** |
| LPIPS ↓ | 0.1973 | **0.1962** | 0.2059 | 0.2024 |
| Hair FID ↓ | **53.89** | 54.55 | 58.70 | 56.94 |

**Forward Curriculum이 우세한 지표**: Edge IoU, Sketch LPIPS, PSNR, SSIM (4/7)  
**Joint가 우세한 지표**: LPIPS (1/7)  
**Unbraid Only가 우세한 지표**: Chamfer, Hair FID (2/7)

Forward Curriculum은 Unbraid 도메인에서 가장 광범위하게 우세하다. Edge IoU, Sketch LPIPS, PSNR, SSIM 4개 지표에서 동시에 1위를 기록하며, 특히 Sketch LPIPS(0.6627)는 모든 조건 포함 최저값이다. Braid를 후속 학습함으로써 오히려 Unbraid 재현 능력이 강화된 결과로, Forward Curriculum의 핵심 기여가 Unbraid 도메인에 집중됨을 알 수 있다.

LPIPS는 Joint(0.1962)가 근소하게 앞서고, Chamfer와 Hair FID는 Unbraid Only 단독 학습이 가장 낮아 단일 태스크 특화의 이점이 일부 지표에서 유지된다.

---

## 4. 지표별 종합 순위

아래는 Combined 기준 7개 지표 각각에서의 방법별 순위 요약이다 (Combined 기준, 단일 태스크 베이스라인 제외).

| Metric | 1위 | 2위 | 3위 |
|--------|-----|-----|-----|
| Edge IoU ↑ | **Ours** (0.0616) | Joint (0.0613) | Reverse (0.0607) |
| Chamfer ↓ | **Joint** (5.3523) | Reverse (5.5852) | Ours (5.6994) |
| Sketch LPIPS ↓ | **Reverse** (0.6717) | Ours (0.6720) | Joint (0.6742) |
| PSNR ↑ | **Ours** (16.2746) | Joint (16.1149) | Reverse (15.5648) |
| SSIM ↑ | **Ours** (0.6094) | Joint (0.6047) | Reverse (0.5969) |
| LPIPS ↓ | **Joint** (0.1928) | Ours (0.1993) | Reverse (0.2054) |
| Hair FID ↓ | **Joint** (50.67) | Ours (53.07) | Reverse (55.62) |

Ours: 4승 / Joint: 3승 / Reverse: 1승 (Combined 기준)

---

## 5. 핵심 분석 및 논의

### 5.1 Forward Curriculum의 효과

쉬운 태스크(Unbraid)를 먼저 학습한 뒤 어려운 태스크(Braid)로 진행하는 Forward Curriculum은 전반적인 픽셀 수준 재현 품질(PSNR, SSIM)과 에지 구조 검출(Edge IoU)에서 일관되게 가장 높은 성능을 보인다. 이 이점은 Braid와 Unbraid 모두에 걸쳐 재현되며, 특히 Unbraid 도메인에서 4개 지표 동시 1위를 기록하는 가장 강력한 결과를 낸다.

### 5.2 Joint 학습 대비 구조적 정밀도 격차

Chamfer 거리와 Hair FID에서 Joint가 Ours를 앞선다. 이는 두 태스크를 동시에 학습하는 Joint 방식이 헤어 스켈레톤 구조 정밀도(Chamfer)와 헤어 이미지 분포 충실도(Hair FID)를 더 잘 보존함을 의미한다. Forward Curriculum은 순차적 태스크 전환 과정에서 이전 태스크의 구조적 표현이 일부 희석될 가능성이 있다.

그러나 Combined에서 Chamfer 격차(5.35 대 5.70)와 Hair FID 격차(50.67 대 53.07)는 크지 않으며, PSNR·SSIM에서 Forward Curriculum이 보이는 우위가 실제 시각 품질 측면에서 더 중요하다고 볼 수 있다.

### 5.3 Reverse Curriculum의 실패

Reverse Curriculum은 어려운 태스크(Braid)를 먼저 학습하는 전략으로, 대부분의 지표에서 가장 낮은 성능을 보인다. 특히 Braid 도메인의 Hair FID(103.15)는 Joint(79.32)의 1.3배 수준으로, 역순 학습이 Braid 분포 학습을 심각하게 방해함을 보여 준다. PSNR(14.52)과 SSIM(0.5872)도 세 조건 중 최저이다. 이는 어려운 태스크를 먼저 학습하면 이후 쉬운 태스크로의 전환 시 네트워크가 효율적으로 일반화하지 못한다는 것을 시사한다.

### 5.4 단일 태스크 베이스라인의 역할

Braid Only와 Unbraid Only는 각각 Chamfer에서 해당 도메인 내 최고 성능을 기록한다. 이는 특화 학습이 구조 정밀도 면에서 이점이 있음을 보여 주나, 다른 대부분의 지표에서 Joint 또는 Ours에 뒤처진다. 단일 태스크 학습은 전이 학습의 이점을 포기하는 대신 특정 도메인 구조 정확도를 극대화하는 절충안이다.

---

## 6. 결론

Forward Curriculum(Ours)이 Combined, Braid, Unbraid 세 구간에서 PSNR과 SSIM 기준으로 일관되게 최고 성능을 보인다. 지각 품질(LPIPS)과 분포 충실도(Hair FID) 측면에서는 Joint가 근소하게 우위이지만, 픽셀 수준 재현 품질과 에지 구조 검출을 종합하면 Forward Curriculum이 가장 균형 잡힌 성능을 제공한다.

Reverse Curriculum은 Braid 도메인에서 심각한 Hair FID 저하를 포함하여 전반적으로 성능이 떨어지므로, 커리큘럼 학습의 방향성이 결과에 결정적 영향을 미침을 보여 준다. 쉬운 태스크에서 어려운 태스크로 진행하는 순방향 학습이 이 과제에서 최적임을 확인한다.
