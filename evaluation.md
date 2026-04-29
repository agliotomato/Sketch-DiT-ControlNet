# Quantitative Evaluation

## Overall Summary (n=107, hair region masked)

| Metric | GAN (SHS) | DiT (full) |
|--------|:---------:|:----------:|
| **Edge IoU ↑** | 0.1031 | **0.1047** |
| **Chamfer Dist ↓** | 2.6926 | **2.6739** |
| **Sketch LPIPS ↓** | **0.7590** | 0.7776 |
| **Hair FID ↓** | 195.06 | **177.63** |
| **LPIPS (GT) ↓** | 0.3263 | **0.3250** |
| **SSIM (GT) ↑** | 0.5906 | **0.5936** |
| **PSNR (GT) ↑** | 11.18 | **12.16** |
| **Boundary FID ↓** | 50.95 | **37.16** |
| **Boundary LPIPS ↓** | 0.0272 | **0.0267** |
| **Face LPIPS ↓** | 0.0036 | **0.0000** |
| **ArcFace Cos ↑** | 0.7697 | **0.7916** |

> FID: 107장 기준 (500장 이하, 참고용). ArcFace Cos: ResNet50 (ImageNet) embedding proxy.

---

## Per-Image Visual Comparison (first 16)

각 행: GT 원본 | Sketch 입력 | GAN (SHS) 결과 | DiT (full) 결과  
각 행 아래: PSNR / SSIM / LPIPS (hair region 내, GAN vs DiT)

---

### braid_2534
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2534.png) | ![](dataset/braid/sketch/test/braid_2534.png) | ![](custom_results/gan/shs/braid_2534.png) | ![](custom_results/test_batch2/braid_2534_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 10.10 | **10.64** |
| SSIM ↑ | 0.560 | **0.562** |
| LPIPS ↓ | 0.357 | **0.343** |

---

### braid_2537
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2537.png) | ![](dataset/braid/sketch/test/braid_2537.png) | ![](custom_results/gan/shs/braid_2537.png) | ![](custom_results/test_batch2/braid_2537_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **8.87** | 7.92 |
| SSIM ↑ | **0.427** | 0.405 |
| LPIPS ↓ | **0.453** | 0.457 |

---

### braid_2539
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2539.png) | ![](dataset/braid/sketch/test/braid_2539.png) | ![](custom_results/gan/shs/braid_2539.png) | ![](custom_results/test_batch2/braid_2539_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 10.03 | **10.42** |
| SSIM ↑ | **0.370** | 0.362 |
| LPIPS ↓ | 0.414 | **0.397** |

---

### braid_2548
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2548.png) | ![](dataset/braid/sketch/test/braid_2548.png) | ![](custom_results/gan/shs/braid_2548.png) | ![](custom_results/test_batch2/braid_2548_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **7.99** | 7.93 |
| SSIM ↑ | 0.675 | **0.680** |
| LPIPS ↓ | 0.285 | **0.285** |

---

### braid_2562
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2562.png) | ![](dataset/braid/sketch/test/braid_2562.png) | ![](custom_results/gan/shs/braid_2562.png) | ![](custom_results/test_batch2/braid_2562_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **12.65** | 12.64 |
| SSIM ↑ | **0.774** | 0.764 |
| LPIPS ↓ | 0.190 | **0.177** |

---

### braid_2572
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2572.png) | ![](dataset/braid/sketch/test/braid_2572.png) | ![](custom_results/gan/shs/braid_2572.png) | ![](custom_results/test_batch2/braid_2572_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 10.03 | **10.84** |
| SSIM ↑ | **0.500** | 0.487 |
| LPIPS ↓ | 0.415 | **0.413** |

---

### braid_2574
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2574.png) | ![](dataset/braid/sketch/test/braid_2574.png) | ![](custom_results/gan/shs/braid_2574.png) | ![](custom_results/test_batch2/braid_2574_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 11.46 | **12.11** |
| SSIM ↑ | **0.722** | 0.717 |
| LPIPS ↓ | **0.302** | 0.318 |

---

### braid_2576
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2576.png) | ![](dataset/braid/sketch/test/braid_2576.png) | ![](custom_results/gan/shs/braid_2576.png) | ![](custom_results/test_batch2/braid_2576_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **10.99** | 10.05 |
| SSIM ↑ | 0.615 | **0.633** |
| LPIPS ↓ | 0.361 | **0.317** |

---

### braid_2590
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2590.png) | ![](dataset/braid/sketch/test/braid_2590.png) | ![](custom_results/gan/shs/braid_2590.png) | ![](custom_results/test_batch2/braid_2590_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 11.07 | **12.40** |
| SSIM ↑ | **0.475** | 0.474 |
| LPIPS ↓ | **0.416** | 0.446 |

---

### braid_2592
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2592.png) | ![](dataset/braid/sketch/test/braid_2592.png) | ![](custom_results/gan/shs/braid_2592.png) | ![](custom_results/test_batch2/braid_2592_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **9.77** | 9.62 |
| SSIM ↑ | 0.436 | **0.451** |
| LPIPS ↓ | **0.450** | 0.476 |

---

### braid_2617
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2617.png) | ![](dataset/braid/sketch/test/braid_2617.png) | ![](custom_results/gan/shs/braid_2617.png) | ![](custom_results/test_batch2/braid_2617_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 12.73 | **13.73** |
| SSIM ↑ | 0.664 | **0.675** |
| LPIPS ↓ | 0.279 | **0.263** |

---

### braid_2625
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2625.png) | ![](dataset/braid/sketch/test/braid_2625.png) | ![](custom_results/gan/shs/braid_2625.png) | ![](custom_results/test_batch2/braid_2625_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 7.61 | **9.80** |
| SSIM ↑ | 0.697 | **0.711** |
| LPIPS ↓ | 0.298 | **0.281** |

---

### braid_2652
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2652.png) | ![](dataset/braid/sketch/test/braid_2652.png) | ![](custom_results/gan/shs/braid_2652.png) | ![](custom_results/test_batch2/braid_2652_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | **10.97** | 10.83 |
| SSIM ↑ | 0.682 | **0.682** |
| LPIPS ↓ | 0.233 | **0.230** |

---

### braid_2653
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2653.png) | ![](dataset/braid/sketch/test/braid_2653.png) | ![](custom_results/gan/shs/braid_2653.png) | ![](custom_results/test_batch2/braid_2653_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 9.83 | **10.23** |
| SSIM ↑ | **0.756** | 0.753 |
| LPIPS ↓ | **0.197** | 0.216 |

---

### braid_2657
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2657.png) | ![](dataset/braid/sketch/test/braid_2657.png) | ![](custom_results/gan/shs/braid_2657.png) | ![](custom_results/test_batch2/braid_2657_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 11.31 | **12.68** |
| SSIM ↑ | **0.579** | 0.579 |
| LPIPS ↓ | 0.316 | **0.311** |

---

### braid_2676
| GT | Sketch | GAN (SHS) | DiT (full) |
|:--:|:------:|:---------:|:----------:|
| ![](dataset/braid/img/test/braid_2676.png) | ![](dataset/braid/sketch/test/braid_2676.png) | ![](custom_results/gan/shs/braid_2676.png) | ![](custom_results/test_batch2/braid_2676_full.png) |

| | GAN | DiT |
|--|:---:|:---:|
| PSNR ↑ | 9.17 | **9.46** |
| SSIM ↑ | **0.775** | 0.771 |
| LPIPS ↓ | 0.192 | **0.192** |
