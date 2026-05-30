

## 현재 Forward 모델 ControlNet 역할 및 Matte 반영 경로

`ctrl_cond`(B, 17, 64×64) 구성
- sketch → VAE encode → `sketch_latent`(B, 16, 64×64)
-  matte → MatteCNN → `matte_feat`(B, 16, 64×64)
- 두 개의 16ch을 element-wise add `fused_feat` (B, 16, 64×64)
- matte를 bilinear downsample하여 `matte_mask`(B, 1, 64×64)
- `fused_feat` 와 `matte_mask` channel-wise concat하여 생성`ctrl_cond`(B,17,64×64)

### 1-1. 입력 conditioning 방식: concat이 아닌 additive embedding

`SD3ControlNetModel`에서 `ctrl_cond` (17ch)는 `pos_embed_input`이라는 별도 patch embed layer를 통과한 뒤, noisy_latent tokens에 **더해진다**.

```
noisy_latent (16ch, 64×64) → main pos_embed  → tokens      (B, 1024, 1152)
ctrl_cond    (17ch, 64×64) → pos_embed_input → cond_tokens (B, 1024, 1152)
combined = tokens + cond_tokens              ← additive(concat이 아님)
```

### 1-2. ControlNet 12블록의 정체 및 주입 방식

**구조 초기화**: `SD3ControlNetModel.from_transformer(transformer, num_layers=12, load_weights_from_transformer=True)`로 생성.
- SD3.5-medium의 transformer block 24개 중 **앞 12개 블록**(front)의 가중치를 복사하여 초기화한다.
- 학습 시에는 frozen SD3와 분리되어 동작.

**주입 방식**: ControlNet 12블록 출력(block_samples[0..11]) → frozen SD3 Transformer 전체 24블록에 interval=2로 residual addition:

- ControlNet block k 출력 -> SD3 block 2k,2k+1 두 곳에 element-wise add로 동일하게 주입
- ControlNet block 0 출력(residual) -> SD3 block 0 block 1 에 element-wise add로 동일하게 주입

```python
hidden_states_i += block_samples[i // 2]   # i = 0..23
```

 front/back 선택적 동작은 없으며, 24개 블록 모두 동일하게 영향을 받는다.

### 1-3. Matte가 실제로 반영되는 경로 (전체 흐름)

```
matte (B, 1, 512×512)
├─ MatteCNN (trainable, 3×stride-2 Conv)
│   └─ matte_feat (B, 16, 64×64)
│       └─ sketch_latent + matte_feat → ctrl_cond 앞 16ch
└─ bilinear downsample
    └─ matte_latent (B, 1, 64×64) → ctrl_cond 뒤 1ch
        [SD3ControlNet extra_conditioning_channels=1]

ctrl_cond (17ch)
 → pos_embed_input
 → ControlNet 12 blocks
 → block_samples[0..11]
 → SD3 Transformer 24 blocks 모두에 residual addition

```

## 정리
| 요소 | 위치 | 역할 |
|------|------|------|
| matte → MatteCNN → `matte_feat` (16ch) | `ctrl_cond` 앞 16ch | sketch_latent와 element-wise add → 구조 정보 융합 |
| matte → bilinear downsample → `matte_mask` (1ch) | `ctrl_cond` 뒤 1ch | 마스크 공간 정보를 명시적 채널로 분리 주입 |
| `ctrl_cond` 전체 (17ch) | ControlNet 입력 | 위 두 경로 합산 → ControlNet 12블록 → SD3 24블록에 residual addition |

 네트워크 내부 블록 출력에 곱해지는 실제 gate tensor가 존재하는 것은 아니다.


