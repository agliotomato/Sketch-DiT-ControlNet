# Hair Patch 생성 가이드

## 전체 파이프라인

```
[S2M 스케치] → S2M (SketchHairSalon) → [matte]
                                            │
                         ┌──────────────────┴──────────────────┐
                         ▼                                      ▼
           [S2I 스케치] + matte                    [S2I 스케치] + matte
         → GAN (hairsalon_S2I_braid)             → DiT (our best.pth)
         → hair patch                            → hair patch
         (infer_hairsalon_custom.py)             (infer_custom.py)
                         │                                      │
                         └──────────────┬───────────────────────┘
                                        ▼
                              [face 이미지] + matte
                            → composite.py
                            → composited image
                                        │
                                        ▼
                              [composited] + matte
                            → inpaint_boundary.py
                            → 최종 합성 이미지
```

---

## Step 1. S2M 스케치 준비 (matte 생성용, 공통)

| 항목 | 값 |
|------|-----|
| 해상도 | **512 × 512** |
| 포맷 | Grayscale (1채널) |
| 배경 | **128** (회색) |
| 머리카락 스트로크 | **255** (흰색) |
| 테두리 (out-stroke) | **0** (검정) — 외곽 경계, 일부만 그려도 됨 |

저장 위치:
```
SketchHairSalon/test_img/braid/input_1/파일명.png
```

---

## Step 2. Matte 생성 (공통)

**체크포인트:** `SketchHairSalon/checkpoints/S2M/200_net_G.pth`

```bash
cd ~/hair-dit/SketchHairSalon
mkdir -p results/generated_matte
python3 S2M_test.py braid      # braid 스타일
python3 S2M_test.py            # unbraid 스타일 (기본값)
```

**결과 저장 위치:** `SketchHairSalon/results/generated_matte/파일명.png`

---

## Step 3. S2I 스케치 준비 (hair patch 생성용, 공통)

| 항목 | 값 |
|------|-----|
| 해상도 | 자유 (모델 내부에서 리사이즈) |
| 포맷 | RGB (3채널) |
| 배경 | **검정 (0, 0, 0)** |
| 머리카락 스트로크 | **원하는 머리 색** |

> 스트로크 색 = 생성될 머리 색의 가이드

---

## Step 4-A. Hair Patch 생성 — GAN 방식

**체크포인트:** `checkpoints/hairsalon_S2I_braid/400_net_G.pth`

```bash
cd ~/hair-dit
python scripts/infer_hairsalon_custom.py \
  --sketch <S2I_스케치_경로> \
  --matte  SketchHairSalon/results/generated_matte/<파일명>.png \
  --style  braid \
  --output_dir custom_results/gan/
```

**결과 파일:**
- `custom_results/gan/{stem}_gen.png` — 최종 hair patch
- `custom_results/gan/{stem}_panel.png` — [원본 sketch | 입력 sketch | 생성 결과]

---

## Step 4-B. Hair Patch 생성 — DiT 방식 (우리 모델)

**체크포인트:** `checkpoints/best.pth`
**config:** braid → `configs/phase2_braid.yaml` / unbraid → `configs/phase1_unbraid.yaml`

```bash
cd ~/hair-dit
python scripts/infer_custom.py \
  --sketch     <S2I_스케치_경로> \
  --matte      SketchHairSalon/results/generated_matte/<파일명>.png \
  --checkpoint checkpoints/dit/phase2_braid/final.pth \
  --config     configs/phase2_braid.yaml \
  --output_dir custom_results/dit/
```

**결과 파일:**
- `custom_results/dit/{stem}_gen.png` — 생성된 hair patch
- `custom_results/dit/{stem}_panel.png` — [sketch | matte | generated]

---

## Step 5. Face 이미지에 합성 (Composite)

생성된 hair patch를 원본 face 이미지에 alpha 합성한다.

```bash
cd ~/hair-dit
python scripts/composite.py \
  --hair   custom_results/dit/<stem>_gen.png \
  --face   <face_이미지_경로> \
  --matte  SketchHairSalon/results/generated_matte/<파일명>.png \
  --output results/composite/<stem>_composited.png
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--feather` | 3.0 | Gaussian feathering sigma. 경계 부드러움 조절 (0=off) |
| `--scale` | 1.0 | hair/matte 스케일 (1.0=원본) |
| `--offset_x` | 0 | x축 이동 (픽셀, 양수=오른쪽) |
| `--offset_y` | 0 | y축 이동 (픽셀, 양수=아래쪽) |

**결과 파일:** `results/composite/{stem}_composited.png`

---

## Step 6. 경계 Inpainting (Boundary Inpainting)

합성 이미지의 hair-face 경계를 SD2 inpainting으로 자연스럽게 blending한다.
경계 안팎 ring(`dilate(matte) - erode(matte)`)을 마스크로 사용하여 양쪽 모두 inpaint.

```bash
cd ~/hair-dit
python scripts/inpaint_boundary.py \
  --composited results/composite/<stem>_composited.png \
  --matte      SketchHairSalon/results/generated_matte/<파일명>.png \
  --output     results/final/<stem>_final.png
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--dilate` | 20 | 경계 ring 두께 (px). 클수록 더 넓은 영역 inpaint |
| `--strength` | 0.45 | inpainting strength. 낮을수록 원본 보존 |
| `--steps` | 20 | denoising steps |

**결과 파일:** `results/final/{stem}_final.png`

> Hair 내부는 항상 원본 composite로 복원되므로 생성된 hair 품질은 유지됨.

---

## 자동화 (한 번에 실행)

`scripts/run_custom_pipeline.sh`:

```bash
#!/bin/bash
# Usage: bash scripts/run_custom_pipeline.sh <s2m_sketch> <s2i_sketch> [braid|unbraid]

S2M_SKETCH=$1
S2I_SKETCH=$2
STYLE=${3:-braid}
STEM=$(basename "$S2M_SKETCH" .png)

# Step 1: matte 생성
cd ~/hair-dit/SketchHairSalon
cp "$S2M_SKETCH" "test_img/${STYLE}/input_1/"
mkdir -p results/generated_matte
python3 S2M_test.py $STYLE

cd ~/hair-dit

# Step 2: GAN
python scripts/infer_hairsalon_custom.py \
  --sketch "$S2I_SKETCH" \
  --matte  "SketchHairSalon/results/generated_matte/${STEM}.png" \
  --style  "$STYLE" \
  --output_dir custom_results/gan/

# Step 3: DiT
python scripts/infer_custom.py \
  --sketch     "$S2I_SKETCH" \
  --matte      "SketchHairSalon/results/generated_matte/${STEM}.png" \
  --checkpoint checkpoints/best.pth \
  --config     configs/phase2_braid.yaml \
  --output_dir custom_results/dit/

echo "완료: custom_results/gan/ 및 custom_results/dit/"
```

실행:
```bash
bash scripts/run_custom_pipeline.sh \
  SketchHairSalon/test_img/braid/input_1/my_hair.png \
  my_s2i_sketch.png \
  braid
```
