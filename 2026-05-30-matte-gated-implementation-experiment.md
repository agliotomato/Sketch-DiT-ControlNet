# [지침] Matte-Gated Residual Schedule — 구현 + 실험 지침 (서현택)

**작성일**: 2026-05-30
**작성**: GenAI 지도(P2) / 교신저자 김동준
**대상**: 서현택 (1저자)
**근거 문서**: `[0530][서현택]matte.md` (현재 구조 확인 완료)
**관련 논문**: `papers/2026-neurips-hair-dit/` (PG 2026)

> ⚠️ **이 지침은 같은 날 먼저 작성한 `2026-05-30-matte-conditioned-correction-pg-ablation.md`(= "gating 하지 말 것")를 대체합니다.** 논의 끝에 **matte-gated를 실제 architectural contribution으로 구현**하는 방향으로 결정했습니다.

---

## 약어·개념 정리

| 약어 | 풀네임 / 의미 |
|------|--------------|
| DiT | Diffusion Transformer (SD3.5-medium = 24 block) |
| ControlNet | sketch+matte 입력을 받아 residual을 만드는 보조망 (12 block) |
| residual | ControlNet block 출력 → DiT block에 더하는 텐서 (`block_samples[k]`) |
| matte-conditioned | matte를 ControlNet **입력**에 넣는 현재 방식 (유지) |
| matte-gated | residual에 matte를 **곱하는** 새 메커니즘 (이번에 구현) |
| matte_tokens | matte를 token 격자(32×32=1024)로 만든 0~1 게이트 |
| schedule | gating을 어느 block에 적용할지 (none/front/all/back) |
| token grid | 512² → latent 64² → patch2 → **32×32 = 1024 token**, hidden 1152 |
| PG | Pacific Graphics 2026 (Abstract 6/1, Paper 6/8 AoE) |
| LPIPS / FID | 지각 거리 / 분포 거리 지표 |

---

## 0. 목표 한 줄

현재 **matte-conditioned**(matte를 입력으로 넣는) 구조 **위에**, ControlNet residual을 matte로 곱하는 **matte-gated residual schedule**을 추가하고, **어느 block에 게이팅하는지(none/front/all/back)** 가 성능을 좌우함을 ablation으로 입증한다 → 논문의 architectural contribution 확립.

---

## 1. 핵심 설계 결정 (반드시 이대로)

### (1) 게이팅은 입력 conditioning "위에" 추가 (제거 아님)
- matte_feat·matte_mask 입력은 **그대로 유지**. 그 위에 residual gate를 얹는다.
- 이렇게 하면 **현재 학습된 모델 = `schedule="none"` 베이스라인**으로 그대로 쓸 수 있음 (재학습 1회 절약).
- → schedule ablation은 "입력 conditioning 고정 + gate schedule만 변경"이 되어 **schedule 효과를 깨끗이 분리**.

### (2) 게이팅은 학습·추론 **양쪽 모두** 적용
- 추론에만 켜면 모델이 게이팅에 적응 못 해 무너짐. **학습 때부터 같은 schedule로 학습**해야 공정.
- 즉 schedule 값마다 **별도 학습 모델**이 필요 (none 제외).

### (3) gate 곱은 token 공간에서, **patchify와 동일한 순서**로
- 가장 흔한 버그: matte를 flatten하는 순서가 SD3 patch embed의 token 순서와 어긋남 → hair/배경이 뒤집혀 곱해짐.
- SD3 patchify는 `Conv2d(patch=2)` → `flatten(2).transpose(1,2)`. matte도 **반드시 같은 순서**로 만들 것 (아래 코드).

---

## 2. 구현 단계

### Step 1 — matte_tokens 준비 (forward 진입부, 1회)

```python
import torch.nn.functional as F

# matte: (B, 1, 512, 512), 값 0~1
# latent 64×64 → patch2 → token 32×32. 따라서 matte도 32×32로.
matte_tok = F.interpolate(matte, size=(32, 32), mode="bilinear", align_corners=False)
matte_tok = matte_tok.flatten(2).transpose(1, 2)   # (B, 1024, 1)  ← patchify와 동일 순서
# hidden은 (B, 1024, 1152). 곱할 때 broadcast: (B,1024,1) * (B,1024,1152) OK
```

> ✅ **검증 필수**: `matte_tok`을 32×32로 reshape해서 시각화 → 원본 matte와 좌우/상하 일치하는지 **눈으로 확인**하고 시작하세요. 여기서 틀리면 전부 무효입니다.

### Step 2 — schedule 함수

```python
def gate_residual(residual, matte_tok, block_idx, schedule):
    # residual: (B, 1024, 1152), matte_tok: (B, 1024, 1)
    apply = {
        "none":       False,
        "front_only": block_idx < 12,
        "all":        True,
        "back_only":  block_idx >= 12,
    }[schedule]
    return residual * matte_tok if apply else residual
```

### Step 3 — 주입 루프 수정 (현재 `hidden_i += block_samples[i//2]` 자리)

diffusers `SD3Transformer2DModel.forward`의 controlnet residual 더하는 부분(대략):

```python
for index_block, block in enumerate(self.transformer_blocks):   # 24 blocks
    encoder_hidden_states, hidden_states = block(hidden_states, ...)
    if controlnet_block_samples is not None:
        interval = len(self.transformer_blocks) // len(controlnet_block_samples)  # = 2
        res = controlnet_block_samples[index_block // interval]
        # ▼▼ 추가 ▼▼
        res = gate_residual(res, matte_tok, index_block, schedule)
        # ▲▲ 추가 ▲▲
        hidden_states = hidden_states + res
```

### Step 4 — plumbing
- `matte_tok`과 `schedule` 문자열을 transformer forward까지 전달 (인자 추가 또는 모듈 attribute로 stash).
- `schedule`은 config/CLI 인자로 빼서 학습 스크립트에서 `--schedule back_only` 식으로 지정 가능하게.

---

## 3. 학습 계획 (PG 일정 — 비용 정직하게)

각 schedule = 별도 모델(phase1 unbraid → phase2 braid). **현실적으로 4개 풀 학습은 D-9 안에 위험**하므로 우선순위를 둡니다.

| 우선 | schedule | 학습 | 비고 |
|:--:|---|---|---|
| ✅ 이미 있음 | `none` | — | **현재 모델** 그대로 = 베이스라인 |
| 🔴 P0 | `back_only` | 신규 1개 | **우리의 가설(Ours)** — 이것부터 |
| 🟠 P1 | `all` | 신규 1개 | 가장 강력한 경쟁 schedule (back vs all 대비가 핵심 메시지) |
| 🟡 P2 | `front_only` | 신규 1개 | 시간 남으면 (대조 완성용) |

- **최소 출판 가능 구성 = none + back_only + all (3개)**. front_only는 못 하면 본문에서 "생략" 처리 가능.
- 학습 비용이 일정상 빠듯하면 **phase2(braid finetune)만 schedule 적용**하고 phase1은 공유하는 단축안도 가능 — 단 이 경우 논문에 "phase2부터 gating 적용"으로 정확히 기술. **결정 전 지도와 상의.**

---

## 4. Ablation 설계 (메인 표)

### 표 1 — Schedule Ablation (핵심 contribution 입증)

입력 conditioning(matte_feat+matte_mask) **고정**, gate schedule만 변경:

| schedule | gate 적용 block | 기대 |
|---|---|---|
| none (현 모델) | 없음 | 베이스라인 |
| front_only | 0–11 | 전역 단계만 제약 → 비효율 예상 |
| all | 0–23 | 전 단계 제약 → 전역 맥락 손상 우려 |
| **back_only (Ours)** | 12–23 | **전역 맥락 보존 + 국소 정제** → 최적 가설 |

**측정 지표 (전부)**: Hair FID, LPIPS(GT), SSIM, PSNR, Edge IoU, Chamfer, **Boundary FID, Boundary LPIPS, Face LPIPS, ArcFace** — 특히 경계/얼굴 4종이 schedule 차이를 가장 잘 드러낼 것.

> **논문 메시지**: "DiT는 모든 block이 전역 attention을 가지므로 U-Net과 달리 앞/뒤 구분이 자동으로 안 생긴다 → schedule로 명시적으로 부여해야 하고, back_only가 전역 구조와 국소 정제를 동시에 만족하는 유일 구성." 이 주장을 표 1이 뒷받침해야 함.

### 표 2 — Matte 자체 기여 (별도, 직교) — `[0529]ablation_list.md` Ablation A 유지
- A1(matte_mask만) / A2(MatteCNN만) / A3(둘다). schedule은 back_only로 고정.
- → "matte를 입력으로 주는 것"과 "gate schedule" 두 축을 분리 입증.

---

## 5. 리스크 & 폴백 (반드시 인지)

| 리스크 | 대응 |
|---|---|
| token 순서 버그 → hair/배경 반전 | **Step 1 시각화 검증 먼저** (가장 흔한 실패) |
| 경계에서 residual을 0으로 끊어 **boundary artifact** 발생 → Boundary LPIPS 악화 | hard 0/1 대신 **soft matte(0~1 그대로)** 사용 (이미 그렇게 설계됨). 그래도 악화 시 gate를 `matte + ε(1-matte)` 형태로 약화 |
| back_only가 none보다 **안 좋게** 나옴 | 솔직하게 보고. 그 경우 §4 표가 "input conditioning만으로 충분"이라는 역결론 → contribution은 1.2(SOTA)로 무게이동, gating은 ablation evidence로 강등 |
| 학습이 D-9 내 안 끝남 | 최소구성(none+back+all)만, phase2-only gating 단축안 |

> **중요**: back_only가 none을 못 이겨도 **그 결과 자체가 논문에 쓸 수 있는 정직한 발견**입니다. 무리해서 좋게 보이게 만들지 마세요.

---

## 6. 일정

| 날짜 | 항목 |
|---|---|
| 5/30 | Step 1~4 구현 + **token 시각화 검증** + back_only 학습 시작 |
| 5/31 | back_only 측정 + all 학습 시작 |
| 6/1 (Abstract D-day) | back_only vs none 결과로 Abstract 반영 (있는 만큼만) |
| 6/2~6/5 | all (+ front_only) 측정, 표 1 완성 |
| 6/6~6/7 | figure/정성 패널 |
| 6/8 (Paper D-day) | 제출 |

---

## 7. 보고 양식 (서현택 → 지도)

각 schedule 학습 끝날 때마다:
1. `schedule` 값 + 학습 설정(phase1 공유 여부, epoch)
2. 11개 지표 표 (none과 같은 줄에 비교)
3. token 시각화 검증 스크린샷 (최초 1회)
4. 정성 비교 패널 (none vs back_only, 경계 부분 확대)

---

## 8. 막히면 즉시 보고할 것

- token 순서 검증이 안 맞으면 → 진행 멈추고 바로 공유.
- 학습 1개 시간(시계 기준)이 얼마나 걸리는지 5/30 중 알려주면 → 4-arm 다 할지 3-arm으로 줄일지 같이 결정.

---

*GenAI 지도(P2) — 2026-05-30*
