## File Structure

```
src/
  models/
    vae_wrapper.py        # SD3.5 VAE (16ch, frozen)
    controlnet_sd35.py    # HairControlNet + MatteCNN
  data/
    dataset.py            # HairRegionDataset (sketch, matte, target triplets)
    augmentation.py       # 4종 augmentation pipeline
    utils.py              # soft_composite, resize_matte_to_latent
  training/
    trainer.py            # Unified trainer (Phase 1 & 2)
    losses.py             # FlowMatchingLoss + PerceptualLoss + SketchEdgeAlignmentLoss
    ema.py                # EMA for HairControlNet

configs/
  base.yaml               # 공통 설정 (model_id, mixed_precision 등)
  phase1_unbraid.yaml     # Phase 1 config
  phase2_braid.yaml       # Phase 2 config

scripts/
  train.py                # python scripts/train.py --config configs/phase1_unbraid.yaml
  evaluate.py             # SHR, MCS, LPIPS, BSS
  infer.py                # single sample inference
  composite.py            # hair region → face image 합성
  check_dataset.py        # dataset triplet 검증
```
