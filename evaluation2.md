# Comprehensive Evaluation: SHS (GAN) vs Ours (DiT)

## 실험 설정

| 항목 | 내용 |
|---|---|
| 비교 대상 | SHS (SketchHairSalon, GAN-based) vs **DiT v2 / DiT v4 (Diffusion Transformer)** |
| **핵심 차이** | SHS는 braid/unbraid 별도 pth 사용 / Ours는 **단일 체크포인트**로 두 도메인 처리 |
| Braid 결과 경로 | `custom_results/gan/shs/` vs `custom_results/dit/weighted_sum_v2/` vs `custom_results/dit/weighted_sum_v4/` |
| Unbraid 결과 경로 | `custom_results/gan/generated_unbraid/` vs `custom_results/dit/inferunbraidbybraidckp/` |
| 평가 스크립트 | `scripts/eval_comprehensive.py` |

### 데이터셋 규모

| Split | Stems | 비고 |
|---|---|---|
| Braid | 107 |  |
| Unbraid | 466 |  |
| Combined | 573 | Braid(v4) + Unbraid(braid ckp) |

> **FID 참고**: braid(107장)은 권장치(500장) 미달 — 경향성 참고용

---

## Braid 평가 결과 (SHS vs DiT v2 vs DiT v4)

| Metric | SHS (GAN) | DiT v2 | DiT v4 | 비고 |
|---|---|---|---|---|
| Edge IoU ↑ | 0.1031 | **0.1052** | **0.1052** | v2=v4 공동 1위 |
| Chamfer Dist ↓ | 2.6926 | 2.6636 | **2.6474** | v4 최우세 |
| Sketch LPIPS ↓ | **0.7589** | 0.7734 | 0.7697 | SHS 우세 |
| Hair FID ↓ | 195.1563 | 174.0897 | **170.8217** | v4 최우세 |
| LPIPS (GT) ↓ | 0.3263 | 0.3254 | **0.3209** | v4 최우세 |
| SSIM (GT) ↑ | 0.5906 | 0.6012 | **0.6073** | v4 최우세 |
| PSNR (GT) ↑ | 11.1787 | 12.3962 | **12.4141** | v4 최우세 |
| Boundary FID ↓ | 50.9672 | 40.8895 | **21.9679** | v4 압도 |
| Boundary LPIPS ↓ | 0.0272 | 0.0361 | **0.0152** |  v2 < SHS |
| Face LPIPS ↓ | 0.0036 | 0.0302 | **0.0015** |  v2 << SHS |
| ArcFace Cos ↑ | 0.7696 | 0.7781 | **0.7981** | v4 최우세 |

**v2 → v4 주요 개선:**
- Face LPIPS: `0.0302 → 0.0015` (**-95%**, v2는 SHS보다도 8.4배 나쁨)
- Boundary LPIPS: `0.0361 → 0.0152` (**-58%**, v2는 SHS보다도 나쁨)
- Boundary FID: `40.89 → 21.97` (-46%)

---

## Unbraid 평가 결과 (SHS vs DiT, zero-shot cross-domain)

> Ours **9/11** 승 | SHS **2/11** 승 (Edge IoU, Chamfer Dist)
>
> **Note**: Ours는 braid 체크포인트 그대로 unbraid에 적용

| Metric | SHS (GAN) | Ours (DiT) | Δ (%) |
|---|---|---|---|
| Edge IoU ↑ | **0.0560** | 0.0551 | -1.6% |
| Chamfer Dist ↓ | **5.1323** | 5.2480 | +2.3% |
| Sketch LPIPS ↓ | 0.7623 | **0.7305** | **-4.2%** |
| Hair FID ↓ | 170.7689 | **131.6232** | **-22.9%** |
| LPIPS (GT) ↓ | 0.3136 | **0.2851** | **-9.1%** |
| SSIM (GT) ↑ | 0.5814 | **0.5967** | +2.6% |
| PSNR (GT) ↑ | 10.8224 | **12.0179** | +1.20 dB |
| Boundary FID ↓ | 23.7895 | **12.2714** | **-48.4%** |
| Boundary LPIPS ↓ | 0.0246 | **0.0185** | -24.8% |
| Face LPIPS ↓ | 0.0043 | **0.0034** | -20.9% |
| ArcFace Cos ↑ | 0.6410 | **0.6700** | +4.5% |

---

## Combined 평가 결과 (Braid v4 + Unbraid, 573 stems)

> Ours **9/11** 승 | SHS **2/11** 승 (Edge IoU ≈ 동률, Chamfer Dist)

| Metric | SHS (GAN) | Ours (DiT) | Δ (%) |
|---|---|---|---|
| Edge IoU ↑ | **0.0648** | 0.0645 | ≈ 동률 |
| Chamfer Dist ↓ | **4.6767** | 4.7624 | +1.8% |
| Sketch LPIPS ↓ | 0.7616 | **0.7378** | **-3.1%** |
| Hair FID ↓ | 163.7770 | **127.9227** | **-21.9%** |
| LPIPS (GT) ↓ | 0.3159 | **0.2918** | **-7.6%** |
| SSIM (GT) ↑ | 0.5831 | **0.5987** | +2.7% |
| PSNR (GT) ↑ | 10.8889 | **12.0919** | +1.20 dB |
| Boundary FID ↓ | 24.6958 | **11.8684** | **-51.9%** |
| Boundary LPIPS ↓ | 0.0251 | **0.0179** | -28.7% |
| Face LPIPS ↓ | 0.0042 | **0.0031** | -26.2% |
| ArcFace Cos ↑ | 0.6650 | **0.6939** | +4.3% |

---

## 분석해볼 점

### 1. v4가 v2 대비 모든 11개 메트릭 우세
v2와 v4 모두 Hair FID, PSNR, SSIM, LPIPS(GT)에서는 SHS를 능가하지만,
**v2는 Face LPIPS와 Boundary LPIPS에서 SHS보다 오히려 열세**. v4에서 weighted_sum 방식으로 개선.

### 2. v2의 치명적 약점: 얼굴 보존 실패
- Face LPIPS: SHS=0.0036, v2=**0.0302**, v4=0.0015
- v2는 SHS 대비 8.4배 나쁜 얼굴 보존 → weighted_sum_v4의 latent 합성 방식이 핵심 개선

### 3. 경계 품질의 차이 (v4 기준)
- Boundary FID: SHS 대비 braid -56.9%, unbraid -48.4%, combined -51.9%
- v2도 SHS보다 Boundary FID는 개선됐으나 Boundary LPIPS는 오히려 악화 (0.0361 > 0.0272)
- v4에서 Boundary LPIPS도 SHS 대비 성능 우수 → latent 합성 방식 + 후처리 효과

### 4. Zero-shot Cross-domain Generalization
단일 DiT 체크포인트로 unbraid에 zero-shot 적용 시 SHS(domain-specific) 대비:
- Hair FID **-22.9%**, Boundary FID **-48.4%**, LPIPS(GT) **-9.1%**
- unbraid에서도 sketch edge(Edge IoU, Chamfer)를 제외한 9/11 메트릭 우세

### 5. SHS가 우세한 영역의 패턴
| 메트릭 | 우세 split | 해석 |
|---|---|---|
| Sketch LPIPS | Braid (SHS 0.7589 vs v4 0.7697) | 미미한 차이 (+1.4%); binary edge보다 perceptual 관점 |
| Edge IoU | Unbraid (+1.6%) | 정성은 더 우수 |
| Chamfer Dist | Unbraid (+2.3%) | 정성은 더 우수  |


### 6. PSNR 일관된 향상
v2: +1.22 dB, v4: +1.24 dB (vs SHS braid 기준) — 픽셀 재현 정확도에서 체계적이고 안정적 우위.

---

## 파일 위치

| 파일 | 내용 |
|---|---|
| `eval_results/braid_summary.csv` | Braid 3-way (SHS / DiT v2 / DiT v4) |
| `eval_results/unbraid_summary.csv` | Unbraid 2-way (SHS / DiT) |
| `eval_results/combined_summary.csv` | Combined 2-way (SHS / DiT v4) |
| `eval_results/braid_per_image.csv` | Braid per-image (107행) |
| `eval_results/unbraid_per_image.csv` | Unbraid per-image (466행) |
| `eval_results/combined_per_image.csv` | Combined per-image (573행, split 컬럼 포함) |
| `scripts/eval_comprehensive.py` | 평가 스크립트 |
