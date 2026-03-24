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
  --checkpoint checkpoints/best.pth \
  --config     configs/phase2_braid.yaml \
  --output_dir custom_results/dit/
```

**결과 파일:**
- `custom_results/dit/{stem}_gen.png` — 생성된 hair patch
- `custom_results/dit/{stem}_panel.png` — [sketch | matte | generated]

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
