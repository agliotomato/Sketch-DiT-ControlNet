# [0528] Roundtrip 결과 — inverse_semi_spatial

> `[hair_orig | sketch_pred | matte | hair_recon]`

- hair_orig : 원본 이미지
- sketch_pred : inverse 모델이 생성한 sketch
- matte : 원본 matte
- hair_recon : sketch_pred를 forward 모델 input으로 주고 생성한 결과

![0000](inverse_semi_spatial/roundtrip_grid.png)


# 이전 결과와 비교.
 - **Sketch Before** (`stage2_check`): DiT-only inverse — DiT feature 32×32에서 stroke 위치 예측, binary mask + grid color sampling 방식. strand-level 공간 정보 소실로 스케치 품질 불량.
- **Sketch After** (`inverse_semi_spatial/last`): CNN + Cross-Attention Fusion 도입 — CNN pixel-space encoder(512→64)로 strand-level 공간 정보 보존, DiT semantic feature를 Cross-Attention(Q=CNN, K/V=DiT)으로 융합. direct RGB prediction으로 mean-collapse 방지.


| GT Sketch | GT Hair | Sketch Before | Hair Before | Sketch After | Hair After |
|-----------|---------|---------------|-------------|--------------|------------|
| ![](dataset/braid/sketch/test/braid_2534.png) | ![](dataset/braid/img/test/braid_2534.png) | ![](stage2_check/braid_2534_sketch.png) | ![](stage2_roundtrip_result/0000_result.png) | ![](inverse_semi_spatial/last/braid_2534_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0000.png) |
| ![](dataset/braid/sketch/test/braid_2537.png) | ![](dataset/braid/img/test/braid_2537.png) | ![](stage2_check/braid_2537_sketch.png) | ![](stage2_roundtrip_result/0001_result.png) | ![](inverse_semi_spatial/last/braid_2537_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0001.png) |
| ![](dataset/braid/sketch/test/braid_2539.png) | ![](dataset/braid/img/test/braid_2539.png) | ![](stage2_check/braid_2539_sketch.png) | ![](stage2_roundtrip_result/0002_result.png) | ![](inverse_semi_spatial/last/braid_2539_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0002.png) |
| ![](dataset/braid/sketch/test/braid_2562.png) | ![](dataset/braid/img/test/braid_2562.png) | ![](stage2_check/braid_2562_sketch.png) | ![](stage2_roundtrip_result/0004_result.png) | ![](inverse_semi_spatial/last/braid_2562_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0004.png) |
| ![](dataset/braid/sketch/test/braid_2572.png) | ![](dataset/braid/img/test/braid_2572.png) | ![](stage2_check/braid_2572_sketch.png) | ![](stage2_roundtrip_result/0005_result.png) | ![](inverse_semi_spatial/last/braid_2572_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0005.png) |


# Forward 모델과의 비교

- **Hair\_Forward** (`results/DiT(before)`): GT sketch를 입력으로 forward ControlNet만 적용한 결과
- **Sketch\_After** (`inverse_semi_spatial/last`): SemiSpatialNet(CNN+Cross-Attn)으로 hair → sketch 변환
- **Hair\_After** (`inverse_semi_spatial/roundtrip_onlyhair`): inverse sketch를 forward에 통과시킨 roundtrip 결과

| GT Sketch | GT Hair | Hair\_Forward (GT→Forward) | Sketch\_After (Inverse) | Hair\_After (Roundtrip) |
|-----------|---------|---------------------------|------------------------|------------------------|
| ![](dataset/braid/sketch/test/braid_2534.png) | ![](dataset/braid/img/test/braid_2534.png) | ![](results/DiT(before)/0000_gen.png) | ![](inverse_semi_spatial/last/braid_2534_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0000.png) |
| ![](dataset/braid/sketch/test/braid_2537.png) | ![](dataset/braid/img/test/braid_2537.png) | ![](results/DiT(before)/0001_gen.png) | ![](inverse_semi_spatial/last/braid_2537_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0001.png) |
| ![](dataset/braid/sketch/test/braid_2539.png) | ![](dataset/braid/img/test/braid_2539.png) | ![](results/DiT(before)/0002_gen.png) | ![](inverse_semi_spatial/last/braid_2539_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0002.png) |
| ![](dataset/braid/sketch/test/braid_2548.png) | ![](dataset/braid/img/test/braid_2548.png) | ![](results/DiT(before)/0003_gen.png) | ![](inverse_semi_spatial/last/braid_2548_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0003.png) |
| ![](dataset/braid/sketch/test/braid_2562.png) | ![](dataset/braid/img/test/braid_2562.png) | ![](results/DiT(before)/0004_gen.png) | ![](inverse_semi_spatial/last/braid_2562_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0004.png) |
| ![](dataset/braid/sketch/test/braid_2572.png) | ![](dataset/braid/img/test/braid_2572.png) | ![](results/DiT(before)/0005_gen.png) | ![](inverse_semi_spatial/last/braid_2572_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0005.png) |
| ![](dataset/braid/sketch/test/braid_2574.png) | ![](dataset/braid/img/test/braid_2574.png) | ![](results/DiT(before)/0006_gen.png) | ![](inverse_semi_spatial/last/braid_2574_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0006.png) |
| ![](dataset/braid/sketch/test/braid_2576.png) | ![](dataset/braid/img/test/braid_2576.png) | ![](results/DiT(before)/0007_gen.png) | ![](inverse_semi_spatial/last/braid_2576_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0007.png) |
| ![](dataset/braid/sketch/test/braid_2590.png) | ![](dataset/braid/img/test/braid_2590.png) | ![](results/DiT(before)/0008_gen.png) | ![](inverse_semi_spatial/last/braid_2590_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0008.png) |
| ![](dataset/braid/sketch/test/braid_2592.png) | ![](dataset/braid/img/test/braid_2592.png) | ![](results/DiT(before)/0009_gen.png) | ![](inverse_semi_spatial/last/braid_2592_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0009.png) |
| ![](dataset/braid/sketch/test/braid_2617.png) | ![](dataset/braid/img/test/braid_2617.png) | ![](results/DiT(before)/0010_gen.png) | ![](inverse_semi_spatial/last/braid_2617_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0010.png) |
| ![](dataset/braid/sketch/test/braid_2625.png) | ![](dataset/braid/img/test/braid_2625.png) | ![](results/DiT(before)/0011_gen.png) | ![](inverse_semi_spatial/last/braid_2625_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0011.png) |
| ![](dataset/braid/sketch/test/braid_2652.png) | ![](dataset/braid/img/test/braid_2652.png) | ![](results/DiT(before)/0012_gen.png) | ![](inverse_semi_spatial/last/braid_2652_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0012.png) |
| ![](dataset/braid/sketch/test/braid_2653.png) | ![](dataset/braid/img/test/braid_2653.png) | ![](results/DiT(before)/0013_gen.png) | ![](inverse_semi_spatial/last/braid_2653_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0013.png) |
| ![](dataset/braid/sketch/test/braid_2657.png) | ![](dataset/braid/img/test/braid_2657.png) | ![](results/DiT(before)/0014_gen.png) | ![](inverse_semi_spatial/last/braid_2657_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0014.png) |
| ![](dataset/braid/sketch/test/braid_2676.png) | ![](dataset/braid/img/test/braid_2676.png) | ![](results/DiT(before)/0015_gen.png) | ![](inverse_semi_spatial/last/braid_2676_sketch.png) | ![](inverse_semi_spatial/roundtrip_onlyhair/0015.png) |
