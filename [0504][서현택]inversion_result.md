# Inversion Adapter 추론 결과

> **모델**: `checkpoints/inversion/best.pth` (epoch 25, val_loss 0.6838)
>
> **추론 결과 경로**: `custom_results/img2sketch/`
> - sketch_pred: `{stem}_sketch.png`
> - matte_pred:  `{stem}_matte.png`

---

## 표 1.hair2sketch/hair2matte 결과 (16 샘플)

| # | Hair GT | sketch_pred | sketch_GT | matte_pred | matte_GT |
|---|---------|-------------|-----------|------------|----------|
| 1 | ![](dataset/braid/img/test/braid_2534.png) | ![](custom_results/img2sketch/braid_2534_sketch.png) | ![](dataset/braid/sketch/test/braid_2534.png) | ![](custom_results/img2sketch/braid_2534_matte.png) | ![](dataset/braid/matte/test/braid_2534.png) |
| 2 | ![](dataset/braid/img/test/braid_2537.png) | ![](custom_results/img2sketch/braid_2537_sketch.png) | ![](dataset/braid/sketch/test/braid_2537.png) | ![](custom_results/img2sketch/braid_2537_matte.png) | ![](dataset/braid/matte/test/braid_2537.png) |
| 3 | ![](dataset/braid/img/test/braid_2548.png) | ![](custom_results/img2sketch/braid_2548_sketch.png) | ![](dataset/braid/sketch/test/braid_2548.png) | ![](custom_results/img2sketch/braid_2548_matte.png) | ![](dataset/braid/matte/test/braid_2548.png) |
| 4 | ![](dataset/braid/img/test/braid_2562.png) | ![](custom_results/img2sketch/braid_2562_sketch.png) | ![](dataset/braid/sketch/test/braid_2562.png) | ![](custom_results/img2sketch/braid_2562_matte.png) | ![](dataset/braid/matte/test/braid_2562.png) |
| 5 | ![](dataset/braid/img/test/braid_2572.png) | ![](custom_results/img2sketch/braid_2572_sketch.png) | ![](dataset/braid/sketch/test/braid_2572.png) | ![](custom_results/img2sketch/braid_2572_matte.png) | ![](dataset/braid/matte/test/braid_2572.png) |
| 6 | ![](dataset/braid/img/test/braid_2574.png) | ![](custom_results/img2sketch/braid_2574_sketch.png) | ![](dataset/braid/sketch/test/braid_2574.png) | ![](custom_results/img2sketch/braid_2574_matte.png) | ![](dataset/braid/matte/test/braid_2574.png) |
| 7 | ![](dataset/braid/img/test/braid_2576.png) | ![](custom_results/img2sketch/braid_2576_sketch.png) | ![](dataset/braid/sketch/test/braid_2576.png) | ![](custom_results/img2sketch/braid_2576_matte.png) | ![](dataset/braid/matte/test/braid_2576.png) |
| 8 | ![](dataset/braid/img/test/braid_2590.png) | ![](custom_results/img2sketch/braid_2590_sketch.png) | ![](dataset/braid/sketch/test/braid_2590.png) | ![](custom_results/img2sketch/braid_2590_matte.png) | ![](dataset/braid/matte/test/braid_2590.png) |
| 9 | ![](dataset/braid/img/test/braid_2592.png) | ![](custom_results/img2sketch/braid_2592_sketch.png) | ![](dataset/braid/sketch/test/braid_2592.png) | ![](custom_results/img2sketch/braid_2592_matte.png) | ![](dataset/braid/matte/test/braid_2592.png) |
| 10 | ![](dataset/braid/img/test/braid_2617.png) | ![](custom_results/img2sketch/braid_2617_sketch.png) | ![](dataset/braid/sketch/test/braid_2617.png) | ![](custom_results/img2sketch/braid_2617_matte.png) | ![](dataset/braid/matte/test/braid_2617.png) |
| 11 | ![](dataset/braid/img/test/braid_2625.png) | ![](custom_results/img2sketch/braid_2625_sketch.png) | ![](dataset/braid/sketch/test/braid_2625.png) | ![](custom_results/img2sketch/braid_2625_matte.png) | ![](dataset/braid/matte/test/braid_2625.png) |
| 12 | ![](dataset/braid/img/test/braid_2652.png) | ![](custom_results/img2sketch/braid_2652_sketch.png) | ![](dataset/braid/sketch/test/braid_2652.png) | ![](custom_results/img2sketch/braid_2652_matte.png) | ![](dataset/braid/matte/test/braid_2652.png) |
| 13 | ![](dataset/braid/img/test/braid_2653.png) | ![](custom_results/img2sketch/braid_2653_sketch.png) | ![](dataset/braid/sketch/test/braid_2653.png) | ![](custom_results/img2sketch/braid_2653_matte.png) | ![](dataset/braid/matte/test/braid_2653.png) |
| 14 | ![](dataset/braid/img/test/braid_2657.png) | ![](custom_results/img2sketch/braid_2657_sketch.png) | ![](dataset/braid/sketch/test/braid_2657.png) | ![](custom_results/img2sketch/braid_2657_matte.png) | ![](dataset/braid/matte/test/braid_2657.png) |
| 15 | ![](dataset/braid/img/test/braid_2676.png) | ![](custom_results/img2sketch/braid_2676_sketch.png) | ![](dataset/braid/sketch/test/braid_2676.png) | ![](custom_results/img2sketch/braid_2676_matte.png) | ![](dataset/braid/matte/test/braid_2676.png) |
| 16 | ![](dataset/braid/img/test/braid_2691.png) | ![](custom_results/img2sketch/braid_2691_sketch.png) | ![](dataset/braid/sketch/test/braid_2691.png) | ![](custom_results/img2sketch/braid_2691_matte.png) | ![](dataset/braid/matte/test/braid_2691.png) |

---

## 표 2. hair2img, hair2sketch 추론 결과

**파이프라인**:
```
Hair GT
  → [HairInversionAdapter]
  → sketch_pred + matte_pred          (custom_results/img2sketch/)
  → [HairControlNet + SD3.5]
  → round-trip 재생성 (full image)    (custom_results/roundtrips/)
```
- `sketch_pred` / `matte_pred`를 그대로 forward model(HairControlNet)에 넣어 hair를 재생성
- face latent와 latent space에서 합성 후 pixel space soft blend
- GT와 round-trip 결과가 유사할수록 inversion quality가 높음

| # | sketch_pred | matte_pred | round-trip | Hair GT |
|---|-------------|------------|------------|---------|
| 1 | ![](custom_results/img2sketch/braid_2534_sketch.png) | ![](custom_results/img2sketch/braid_2534_matte.png) | ![](custom_results/roundtrips/braid_2534_full.png) | ![](dataset/braid/img/test/braid_2534.png) |
| 2 | ![](custom_results/img2sketch/braid_2537_sketch.png) | ![](custom_results/img2sketch/braid_2537_matte.png) | ![](custom_results/roundtrips/braid_2537_full.png) | ![](dataset/braid/img/test/braid_2537.png) |
| 3 | ![](custom_results/img2sketch/braid_2548_sketch.png) | ![](custom_results/img2sketch/braid_2548_matte.png) | ![](custom_results/roundtrips/braid_2548_full.png) | ![](dataset/braid/img/test/braid_2548.png) |
| 4 | ![](custom_results/img2sketch/braid_2562_sketch.png) | ![](custom_results/img2sketch/braid_2562_matte.png) | ![](custom_results/roundtrips/braid_2562_full.png) | ![](dataset/braid/img/test/braid_2562.png) |
| 5 | ![](custom_results/img2sketch/braid_2572_sketch.png) | ![](custom_results/img2sketch/braid_2572_matte.png) | ![](custom_results/roundtrips/braid_2572_full.png) | ![](dataset/braid/img/test/braid_2572.png) |
| 6 | ![](custom_results/img2sketch/braid_2574_sketch.png) | ![](custom_results/img2sketch/braid_2574_matte.png) | ![](custom_results/roundtrips/braid_2574_full.png) | ![](dataset/braid/img/test/braid_2574.png) |
| 7 | ![](custom_results/img2sketch/braid_2576_sketch.png) | ![](custom_results/img2sketch/braid_2576_matte.png) | ![](custom_results/roundtrips/braid_2576_full.png) | ![](dataset/braid/img/test/braid_2576.png) |
| 8 | ![](custom_results/img2sketch/braid_2590_sketch.png) | ![](custom_results/img2sketch/braid_2590_matte.png) | ![](custom_results/roundtrips/braid_2590_full.png) | ![](dataset/braid/img/test/braid_2590.png) |
| 9 | ![](custom_results/img2sketch/braid_2592_sketch.png) | ![](custom_results/img2sketch/braid_2592_matte.png) | ![](custom_results/roundtrips/braid_2592_full.png) | ![](dataset/braid/img/test/braid_2592.png) |
| 10 | ![](custom_results/img2sketch/braid_2617_sketch.png) | ![](custom_results/img2sketch/braid_2617_matte.png) | ![](custom_results/roundtrips/braid_2617_full.png) | ![](dataset/braid/img/test/braid_2617.png) |
| 11 | ![](custom_results/img2sketch/braid_2625_sketch.png) | ![](custom_results/img2sketch/braid_2625_matte.png) | ![](custom_results/roundtrips/braid_2625_full.png) | ![](dataset/braid/img/test/braid_2625.png) |
| 12 | ![](custom_results/img2sketch/braid_2652_sketch.png) | ![](custom_results/img2sketch/braid_2652_matte.png) | ![](custom_results/roundtrips/braid_2652_full.png) | ![](dataset/braid/img/test/braid_2652.png) |
| 13 | ![](custom_results/img2sketch/braid_2653_sketch.png) | ![](custom_results/img2sketch/braid_2653_matte.png) | ![](custom_results/roundtrips/braid_2653_full.png) | ![](dataset/braid/img/test/braid_2653.png) |
| 14 | ![](custom_results/img2sketch/braid_2657_sketch.png) | ![](custom_results/img2sketch/braid_2657_matte.png) | ![](custom_results/roundtrips/braid_2657_full.png) | ![](dataset/braid/img/test/braid_2657.png) |
| 15 | ![](custom_results/img2sketch/braid_2676_sketch.png) | ![](custom_results/img2sketch/braid_2676_matte.png) | ![](custom_results/roundtrips/braid_2676_full.png) | ![](dataset/braid/img/test/braid_2676.png) |
| 16 | ![](custom_results/img2sketch/braid_2691_sketch.png) | ![](custom_results/img2sketch/braid_2691_matte.png) | ![](custom_results/roundtrips/braid_2691_full.png) | ![](dataset/braid/img/test/braid_2691.png) |
