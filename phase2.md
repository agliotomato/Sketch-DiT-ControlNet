# Phase 2: Braid Fine-tuning 결과

## 학습 설정

| 항목 | 값 |
|---|---|
| 백본 | SD3.5 Medium (8B) + HairControlNet (12 layers) |
| Phase 1 체크포인트 | `checkpoints/phase1_unbraid/epoch_40.pth` |
| 데이터셋 | braid_train (1,000샘플) |
| Epochs | 100 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Loss | Flow matching + LPIPS (즉시 활성화) + Edge alignment (w=0.05) |
| 최종 avg loss | 0.0105 |

---

## Inference 결과 (braid_test, 16샘플, 20 steps)


| 열 | 이름 | 설명 |
|---|---|---|
| 1열 | **Sketch** | 입력 컬러 스케치. 색은 strand 구분자(appearance 아님). 땋임 패턴, 선 방향, 교차 구조 포함 |
| 2열 | **Matte** | 입력 soft alpha matte. 머리카락 영역(0~1). 모델이 생성 범위를 이 영역으로 제한 |
| 3열 | **Generated** | 모델 출력. sketch + matte를 조건으로 flow matching 20스텝 denoising하여 생성한 hair region |
| 4열 | **Target** | 정답. 원본 사진에서 `img × matte`로 추출한 실제 머리카락 영역 |

---

## Inference 결과 이미지 (braid_test, 16샘플)

| Sketch | Matte | Generated | Target |
|--------|-------|-----------|--------|
| <img src="dataset/braid/sketch/test/braid_2534.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2534.png" width="256"/> | <img src="results/0000_gen.png" width="256"/> | <img src="results/0000_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2537.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2537.png" width="256"/> | <img src="results/0001_gen.png" width="256"/> | <img src="results/0001_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2539.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2539.png" width="256"/> | <img src="results/0002_gen.png" width="256"/> | <img src="results/0002_gen.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2548.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2548.png" width="256"/> | <img src="results/0003_gen.png" width="256"/> | <img src="results/0003_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2562.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2562.png" width="256"/> | <img src="results/0004_gen.png" width="256"/> | <img src="results/0004_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2572.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2572.png" width="256"/> | <img src="results/0005_gen.png" width="256"/> | <img src="results/0005_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2574.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2574.png" width="256"/> | <img src="results/0006_gen.png" width="256"/> | <img src="results/0006_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2576.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2576.png" width="256"/> | <img src="results/0007_gen.png" width="256"/> | <img src="results/0007_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2590.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2590.png" width="256"/> | <img src="results/0008_gen.png" width="256"/> | <img src="results/0008_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2592.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2592.png" width="256"/> | <img src="results/0009_gen.png" width="256"/> | <img src="results/0009_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2617.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2617.png" width="256"/> | <img src="results/0010_gen.png" width="256"/> | <img src="results/0010_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2625.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2625.png" width="256"/> | <img src="results/0011_gen.png" width="256"/> | <img src="results/0011_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2652.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2652.png" width="256"/> | <img src="results/0012_gen.png" width="256"/> | <img src="results/0012_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2653.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2653.png" width="256"/> | <img src="results/0013_gen.png" width="256"/> | <img src="results/0013_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2657.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2657.png" width="256"/> | <img src="results/0014_gen.png" width="256"/> | <img src="results/0014_target.png" width="256"/> |
| <img src="dataset/braid/sketch/test/braid_2676.png" width="256"/> | <img src="dataset/braid/matte/test/braid_2676.png" width="256"/> | <img src="results/0015_gen.png" width="256"/> | <img src="results/0015_target.png" width="256"/> |

## 다음 단계
- Step 3: diffusion inpainting refinement — 경계 자연스러운 블렌딩 
 




