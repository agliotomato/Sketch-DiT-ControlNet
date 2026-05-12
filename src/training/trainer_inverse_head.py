"""
InverseHeadTrainer: Hair→Sketch via partially-unfrozen DiT + trainable FPN mask decoder.

Backbone:
  blocks 0-11:  frozen base + inverse LoRA (to_q/k/v, rank-8)
  blocks 12-23: shared backbone — fully unfrozen, lower LR via backbone_lr_scale

Loss schedule (Stage 2, bidirectional=False):
  Phase 1 (epoch < cycle_start):
    Every step → supervised (structure BCE + color L1 + TV)
  Phase 2 (epoch >= cycle_start):
    Every step → supervised + cycle (feature MSE + image LPIPS)

Loss schedule (Stage 3, bidirectional=True, controlnet_trainable=True — Option B):
  Alternating per global_step:
    even step → "inverse batch":
        supervised [+ cycle if phase2] → inverse params + ControlNet (via cycle)
    odd  step → "forward batch":
        FM loss (inverse frozen)       → ControlNet only

  Regression monitor: _validate_forward() logs val_loss_fwd every eval_every epochs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import SD3Transformer2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairRegionDataset
from src.models.controlnet_sd35 import HairControlNet
from src.models.inverse_head import HairToSketchDiT
from src.models.vae_wrapper import VAEWrapper
from src.training.losses_inversion import InversionLoss, tv_loss


class InverseHeadTrainer:
    def __init__(self, config: dict):
        self.cfg  = config
        tcfg      = config["training"]

        # tensorboard only — wandb removed to avoid media-API instability
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            log_with=["tensorboard"],
            project_dir=config["checkpointing"]["output_dir"],
        )

        model_id        = config["model"]["model_id"]
        local_only      = config.get("local_files_only", False)
        controlnet_ckpt = config["model"]["controlnet_checkpoint"]

        # --- Frozen VAE ---
        self.vae = VAEWrapper.from_pretrained(
            model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_only,
        ).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # --- DiT backbone: freeze ALL first; DiTFeatureExtractor will selectively
        #     re-enable blocks 12-23 and inject LoRA on blocks 0-11. ---
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=local_only,
        )
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.transformer.eval()

        # --- Frozen forward ControlNet (phase 2 feature cycle + LPIPS cycle) ---
        self.forward_controlnet: HairControlNet | None = None
        self.w_cycle          = tcfg["loss_weights"].get("cycle", 0.01)
        self.w_lpips          = tcfg["loss_weights"].get("lpips_cycle", 0.05)
        self.cycle_start      = tcfg.get("cycle_start", 9999)
        self.use_gt_matte     = config.get("use_gt_matte", False)
        self.bidirectional        = config.get("bidirectional", False)
        self.controlnet_trainable = config.get("controlnet_trainable", False)
        self.w_fm_forward         = config.get("w_fm_forward", 0.1)
        if (self.w_cycle > 0 or self.w_lpips > 0 or self.controlnet_trainable) \
                and Path(controlnet_ckpt).exists():
            fwd_cn = HairControlNet(
                model_id=model_id, vae=self.vae,
                num_layers=config["model"].get("num_controlnet_layers", 12),
                local_files_only=local_only,
            )
            ckpt = torch.load(controlnet_ckpt, map_location="cpu", weights_only=True)
            fwd_cn.load_state_dict(ckpt["controlnet"])
            if self.controlnet_trainable:
                fwd_cn.train()                          # Option B: trainable
            else:
                fwd_cn.eval()
                for p in fwd_cn.parameters():
                    p.requires_grad_(False)             # Option A / Stage 2: frozen
            self.forward_controlnet = fwd_cn

        # --- Null embeddings (reuse from forward ControlNet if available) ---
        if self.forward_controlnet is not None:
            null_enc_hs = self.forward_controlnet.null_encoder_hidden_states.detach()
            null_pooled = self.forward_controlnet.null_pooled_projections.detach()
        else:
            null_enc_hs = torch.zeros(1, 333, 4096, dtype=torch.bfloat16)
            null_pooled = torch.zeros(1, 2048, dtype=torch.bfloat16)
        # Keep references for LPIPS cycle (transformer forward pass with null cond)
        self._null_enc_hs = null_enc_hs
        self._null_pooled = null_pooled

        # --- Trainable model: LoRA (blocks 0-11) + back_blocks (12-23) + FPN decoders ---
        lora_cfg = config.get("lora", {})
        self.model = HairToSketchDiT(
            transformer=self.transformer,
            vae=self.vae,
            null_enc_hs=null_enc_hs,
            null_pooled=null_pooled,
            lora_rank=lora_cfg.get("rank", 8),
            lora_alpha=lora_cfg.get("alpha", 8.0),
            grid_size=config.get("grid_size", 16),
        )

        # --- LPIPS metric (frozen VGG network) ---
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="vgg").eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
        except ImportError:
            self.lpips_fn = None
            self.w_lpips  = 0.0
            self.accelerator.print("WARNING: lpips not installed; image LPIPS cycle disabled.")

        # --- Data ---
        aug = build_augmentation_pipeline("pretrain")
        train_ds = ConcatDataset([
            HairRegionDataset(split=s, augmentation=aug)
            for s in ("unbraid_train", "braid_train")
        ])
        val_ds = ConcatDataset([
            HairRegionDataset(split=s)
            for s in ("unbraid_test", "braid_test")
        ])

        bs = tcfg.get("batch_size", 4)
        self.train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        # --- Optimizer: separate param groups for back_blocks (lower LR) ---
        backbone_lr_scale = config.get("backbone_lr_scale", 0.1)
        base_lr           = tcfg.get("learning_rate", 1e-4)
        back_block_param_ids = {
            id(p) for p in self.model.feature_extractor.back_blocks.parameters()
            if p.requires_grad
        }
        back_params = [
            p for p in self.model.feature_extractor.back_blocks.parameters()
            if p.requires_grad
        ]
        main_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in back_block_param_ids
        ]
        self.optimizer = AdamW(
            [
                {"params": main_params, "lr": base_lr,                       "_base_lr": base_lr},
                {"params": back_params, "lr": base_lr * backbone_lr_scale,   "_base_lr": base_lr * backbone_lr_scale},
            ],
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        total_steps  = tcfg.get("epochs", 50) * len(self.train_loader)
        warmup_steps = tcfg.get("warmup_steps", 200)
        self.warmup_steps = warmup_steps
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6,
        )

        # --- ControlNet optimizer (Option B only) ---
        self.optimizer_cn    = None
        self.lr_scheduler_cn = None
        if self.controlnet_trainable and self.forward_controlnet is not None:
            cn_lr = config.get("controlnet_lr", 5e-6)
            self.optimizer_cn = AdamW(
                [p for p in self.forward_controlnet.parameters() if p.requires_grad],
                lr=cn_lr, betas=(0.9, 0.999), weight_decay=1e-2,
            )
            self.lr_scheduler_cn = CosineAnnealingLR(
                self.optimizer_cn, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-8,
            )

        # --- Loss ---
        lw = tcfg.get("loss_weights", {})
        self.loss_fn = InversionLoss(
            w_structure=lw.get("structure", 1.0),
            w_color=lw.get("color", 0.5),
        )
        self.w_tv = lw.get("tv", 0.01)

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Accelerate prepare ---
        if self.controlnet_trainable and self.forward_controlnet is not None:
            (
                self.model, self.optimizer, self.optimizer_cn,
                self.forward_controlnet,
                self.train_loader, self.val_loader,
                self.lr_scheduler, self.lr_scheduler_cn,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, self.optimizer_cn,
                self.forward_controlnet,
                self.train_loader, self.val_loader,
                self.lr_scheduler, self.lr_scheduler_cn,
            )
        else:
            (
                self.model, self.optimizer,
                self.train_loader, self.val_loader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer,
                self.train_loader, self.val_loader,
                self.lr_scheduler,
            )
        device = self.accelerator.device
        self.vae         = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        if self.forward_controlnet is not None and not self.controlnet_trainable:
            self.forward_controlnet = self.forward_controlnet.to(device)
        if self.lpips_fn is not None:
            self.lpips_fn = self.lpips_fn.to(device)
        self._null_enc_hs = self._null_enc_hs.to(device)
        self._null_pooled = self._null_pooled.to(device)

        run_name = config.get("run_name", "inverse_head_dit_partial_unfreeze")
        self.accelerator.init_trackers(
            project_name=run_name,
            config=_flatten(config),
        )

        self.global_step    = 0
        self.best_val_loss  = float("inf")
        self.start_epoch    = 0
        self._current_epoch = 0
        self._restore_training_state()
        self._setup_stage(config)

    def _restore_training_state(self):
        resume = self.cfg.get("training", {}).get("resume")
        if not resume or not Path(resume).exists():
            return
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model"])
        if "optimizer"    in ckpt: self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt: self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.global_step    = ckpt.get("global_step", 0)
        self.best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        self.start_epoch    = self.cfg["training"].get("start_epoch") or ckpt.get("epoch", 0)
        self.accelerator.print(
            f"Resumed from {resume} (epoch {self.start_epoch}, step {self.global_step})"
        )

    def _setup_stage(self, config: dict) -> None:
        stage = config.get("stage", 2)
        mode  = "Option B (CN trainable)" if self.controlnet_trainable else "Option A (CN frozen)"
        self.accelerator.print(f"[Stage {stage}] {mode} — inverse LoRA + shared back_blocks ready.")

    def _freeze_inverse_params(self) -> None:
        """Forward FM step 중 inverse module 전체 freeze (Option B)."""
        inv = self.accelerator.unwrap_model(self.model)
        inv.feature_extractor.freeze_lora()
        for p in inv.feature_extractor.back_blocks.parameters():
            p.requires_grad_(False)
        for p in inv.stroke_decoder.parameters():
            p.requires_grad_(False)
        inv.feature_extractor.agg_weights.requires_grad_(False)

    def _unfreeze_inverse_params(self) -> None:
        inv = self.accelerator.unwrap_model(self.model)
        inv.feature_extractor.unfreeze_lora()
        for p in inv.feature_extractor.back_blocks.parameters():
            p.requires_grad_(True)
        for p in inv.stroke_decoder.parameters():
            p.requires_grad_(True)
        inv.feature_extractor.agg_weights.requires_grad_(True)

    def train(self):
        tcfg       = self.cfg["training"]
        epochs     = tcfg.get("epochs", 50)
        save_every = self.cfg["checkpointing"].get("save_every", 10)
        eval_every = self.cfg["checkpointing"].get("eval_every", 5)

        for epoch in range(self.start_epoch, epochs):
            self._current_epoch = epoch
            phase2 = (epoch >= self.cycle_start)
            
            if phase2 and epoch == self.cycle_start:
                self.accelerator.print("\n" + "="*50)
                self.accelerator.print("🚀 [Phase 2] 본격적인 Joint 학습 (Supervised + Cycle)을 시작합니다!")
                self.accelerator.print("="*50 + "\n")

            self.model.train()
            epoch_losses = []

            desc = f"Epoch {epoch+1}/{epochs}" + (" [+cycle-alt]" if phase2 else "")
            progress = tqdm(
                self.train_loader, desc=desc,
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress:

                if self.bidirectional and self.controlnet_trainable:
                    # ── Option B: alternating mini-batch ──────────────────────
                    # even step: inverse batch (supervised [+cycle] → inv + CN)
                    # odd  step: forward batch (FM → CN only, inverse frozen)
                    is_fwd_step = (self.global_step % 2 == 1)

                    if is_fwd_step:
                        # Forward FM: CN update only
                        self._freeze_inverse_params()
                        self.optimizer_cn.zero_grad()
                        loss_fm, log_fm = self._forward_fm_step(batch, update_optimizer=False)
                        if self.cfg["training"].get("gradient_clip"):
                            cn_params = [p for p in self.accelerator.unwrap_model(
                                self.forward_controlnet).parameters() if p.requires_grad]
                            self.accelerator.clip_grad_norm_(cn_params, self.cfg["training"]["gradient_clip"])
                        self.optimizer_cn.step()
                        self._unfreeze_inverse_params()
                        loss    = loss_fm
                        log_dict = {
                            "loss_fm_forward": log_fm.get("loss_fm_forward", 0.0),
                            "loss_total":      log_fm["loss_total"],
                            "step_type":       4.0,   # forward-only step
                        }
                    else:
                        # Inverse + cycle: both inv and CN update
                        self.optimizer.zero_grad()
                        self.optimizer_cn.zero_grad()
                        loss_sup, log_sup = self._supervised_step(batch, update_optimizer=False)
                        if phase2:
                            loss_cyc, log_cyc = self._cycle_step(batch, update_optimizer=False)
                        else:
                            loss_cyc = torch.tensor(0.0, device=self.accelerator.device)
                            log_cyc  = {"loss_cycle_feat": 0.0, "loss_cycle_lpips": 0.0, "loss_total": 0.0}
                        if self.cfg["training"].get("gradient_clip"):
                            self.accelerator.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                self.cfg["training"]["gradient_clip"],
                            )
                        self.optimizer.step()
                        self.optimizer_cn.step()
                        loss = loss_sup + loss_cyc
                        log_dict = {
                            "loss_structure":   log_sup.get("loss_structure", 0.0),
                            "loss_color":       log_sup.get("loss_color", 0.0),
                            "loss_tv":          log_sup.get("loss_tv", 0.0),
                            "loss_cycle_feat":  log_cyc.get("loss_cycle_feat", 0.0),
                            "loss_cycle_lpips": log_cyc.get("loss_cycle_lpips", 0.0),
                            "loss_total":       log_sup["loss_total"] + log_cyc["loss_total"],
                            "step_type":        3.0,
                        }

                elif self.bidirectional:
                    # ── Original bidirectional (Option A / back_blocks only) ──
                    self.optimizer.zero_grad()
                    loss_sup, log_sup = self._supervised_step(batch, update_optimizer=False)

                    if phase2:
                        loss_cyc, log_cyc = self._cycle_step(batch, update_optimizer=False)
                    else:
                        loss_cyc = torch.tensor(0.0, device=self.accelerator.device)
                        log_cyc  = {"loss_cycle_feat": 0.0, "loss_cycle_lpips": 0.0}

                    loss_fm, log_fm = self._forward_fm_step(batch, update_optimizer=False)

                    if self.cfg["training"].get("gradient_clip"):
                        self.accelerator.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.cfg["training"]["gradient_clip"],
                        )
                    self.optimizer.step()

                    loss = loss_sup + loss_cyc + loss_fm
                    log_dict = {
                        "loss_structure":   log_sup.get("loss_structure", 0.0),
                        "loss_color":       log_sup.get("loss_color", 0.0),
                        "loss_tv":          log_sup.get("loss_tv", 0.0),
                        "loss_cycle_feat":  log_cyc.get("loss_cycle_feat", 0.0),
                        "loss_cycle_lpips": log_cyc.get("loss_cycle_lpips", 0.0),
                        "loss_fm_forward":  log_fm.get("loss_fm_forward", 0.0),
                        "loss_total":       log_sup["loss_total"] + loss_cyc.item() + log_fm["loss_total"],
                        "step_type":        3.0,
                    }

                elif phase2:
                    loss_sup, log_sup = self._supervised_step(batch, update_optimizer=False)
                    loss_cyc, log_cyc = self._cycle_step(batch, update_optimizer=False)

                    if self.cfg["training"].get("gradient_clip"):
                        self.accelerator.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.cfg["training"]["gradient_clip"],
                        )
                    self.optimizer.step()

                    loss = loss_sup + loss_cyc
                    log_dict = {
                        **log_cyc,
                        "loss_structure": log_sup.get("loss_structure", 0.0),
                        "loss_color":     log_sup.get("loss_color", 0.0),
                        "loss_tv":        log_sup.get("loss_tv", 0.0),
                        "loss_total":     log_sup["loss_total"] + log_cyc["loss_total"],
                        "step_type":      2.0,
                    }

                else:
                    loss, log_dict = self._supervised_step(batch)

                epoch_losses.append(log_dict["loss_total"])

                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg["_base_lr"] * lr_scale
                    if self.optimizer_cn is not None:
                        for pg in self.optimizer_cn.param_groups:
                            pg["lr"] = pg["lr"] * lr_scale
                else:
                    self.lr_scheduler.step()
                    if self.lr_scheduler_cn is not None:
                        self.lr_scheduler_cn.step()

                self.global_step += 1
                progress.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
                self.accelerator.log(log_dict, step=self.global_step)

            avg = sum(epoch_losses) / len(epoch_losses)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

            if (epoch + 1) % eval_every == 0:
                val_loss = self._validate()
                self.accelerator.print(f"Val loss (inv): {val_loss:.4f}")
                log_eval = {"val_loss": val_loss, "epoch": epoch + 1}

                if self.controlnet_trainable:
                    fwd_loss = self._validate_forward()
                    self.accelerator.print(f"Val loss (fwd): {fwd_loss:.4f}  ← regression monitor")
                    log_eval["val_loss_fwd"] = fwd_loss

                self.accelerator.log(log_eval, step=self.global_step)

                # Save BEFORE image logging — logging crashes must not lose checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best.pth", epoch)
                self._save("last.pth", epoch)   # always-overwrite latest

                # Wrap media logging so failures don't kill training
                try:
                    self._log_images(epoch)
                except Exception as e:
                    self.accelerator.print(f"[warn] _log_images failed (skipping): {e}")

            if (epoch + 1) % save_every == 0:
                self._save(f"epoch_{epoch+1}.pth", epoch)

        self._save("final.pth", epochs - 1)
        self.accelerator.end_training()

    def _supervised_step(self, batch: dict, update_optimizer: bool = True) -> tuple[torch.Tensor, dict]:
        """Supervised loss only: structure BCE + color L1 + TV."""
        device = self.accelerator.device
        dtype  = torch.bfloat16

        hair_image = batch["img"].to(device, dtype=dtype)
        sketch_gt  = batch["sketch"].to(device, dtype=dtype)
        matte_gt   = batch["matte"].to(device, dtype=dtype)

        if update_optimizer:
            self.optimizer.zero_grad()

        sketch_pred, stroke_mask = self.model(hair_image, matte_gt)

        loss, log_dict = self.loss_fn(
            stroke_mask_pred=stroke_mask,
            sketch_pred=sketch_pred,
            sketch_gt=sketch_gt,
        )

        l_tv = tv_loss(stroke_mask.float())
        loss = loss + self.w_tv * l_tv
        log_dict["loss_tv"]    = l_tv.item()
        log_dict["loss_total"] = loss.item()
        log_dict["step_type"]  = 0.0

        self.accelerator.backward(loss)
        
        if update_optimizer:
            if self.cfg["training"].get("gradient_clip"):
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg["training"]["gradient_clip"],
                )
            self.optimizer.step()
            
        return loss, log_dict

    def _cycle_step(self, batch: dict, update_optimizer: bool = True) -> tuple[torch.Tensor, dict]:
        """Cycle loss step (phase 2): feature MSE + image-space LPIPS cycle.

        Memory: cycle adds forward_CN forward + DiT 2nd pass + VAE decode + VGG.
        At full batch this OOMs on 40GB GPUs. We split the batch into N micro-batches
        and gradient-accumulate — keeps supervised step at full batch (proven to fit).

        Both losses use enable_grad=True paths so gradients propagate
        back through frozen forward models to the inverse model parameters.
        """
        # If both cycle losses disabled, fall back to supervised step
        feat_active  = self.forward_controlnet is not None and self.w_cycle > 0
        lpips_active = self.w_lpips > 0 and self.lpips_fn is not None
        if not feat_active and not lpips_active:
            return self._supervised_step(batch, update_optimizer=update_optimizer)

        device = self.accelerator.device
        dtype  = torch.bfloat16

        full_hair   = batch["img"].to(device, dtype=dtype)
        full_sketch = batch["sketch"].to(device, dtype=dtype)
        full_matte  = batch["matte"].to(device, dtype=dtype)

        bs        = full_hair.shape[0]
        n_micro   = self.cfg.get("cycle_micro_batches", 2) if bs > 1 else 1
        n_micro   = max(1, min(n_micro, bs))
        chunk_sz  = (bs + n_micro - 1) // n_micro

        if update_optimizer:
            self.optimizer.zero_grad()
        agg_log: dict = {"loss_cycle_feat": 0.0, "loss_cycle_lpips": 0.0, "loss_total": 0.0}
        agg_total: float = 0.0

        for i in range(n_micro):
            s = slice(i * chunk_sz, min((i + 1) * chunk_sz, bs))
            hair_image = full_hair[s]
            sketch_gt  = full_sketch[s]
            matte_gt   = full_matte[s]

            sketch_pred, stroke_mask = self.model(hair_image, matte_gt)

            loss_terms: list[torch.Tensor] = []

            if feat_active:
                block_pred = self.forward_controlnet._get_features_impl(
                    sketch_pred.float(), matte_gt.float(), enable_grad=True,
                )
                with torch.no_grad():
                    block_gt = self.forward_controlnet._get_features_impl(
                        sketch_gt.float(), matte_gt.float(),
                    )
                feat_cycle = sum(
                    F.mse_loss(bp, bg.detach())
                    for bp, bg in zip(block_pred, block_gt)
                ) / len(block_pred)
                loss_terms.append(self.w_cycle * feat_cycle)
                agg_log["loss_cycle_feat"] += feat_cycle.item() / n_micro

            if lpips_active:
                lpips_loss = self._compute_image_lpips_cycle(
                    sketch_pred.float(), matte_gt.float(), hair_image.float(),
                )
                loss_terms.append(self.w_lpips * lpips_loss)
                agg_log["loss_cycle_lpips"] += lpips_loss.item() / n_micro

            micro_total = sum(loss_terms[1:], loss_terms[0])
            self.accelerator.backward(micro_total / n_micro)
            agg_total += micro_total.item() / n_micro

            del sketch_pred, stroke_mask, micro_total, loss_terms
            if feat_active:
                del block_pred, block_gt, feat_cycle
            if lpips_active:
                del lpips_loss

        if update_optimizer:
            if self.cfg["training"].get("gradient_clip"):
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg["training"]["gradient_clip"],
                )
            self.optimizer.step()

        agg_log["loss_total"] = agg_total
        agg_log["step_type"]  = 1.0
        return torch.tensor(agg_total, device=device), agg_log

    def _forward_fm_step(self, batch: dict, update_optimizer: bool = True) -> tuple[torch.Tensor, dict]:
        """Forward flow matching step: sketch_gt + matte_gt → hair (rectified flow).

        Only back_blocks receive gradients — inverse LoRA is frozen during this step.
        ControlNet injection handles front block conditioning (same as v4).
        Requires forward_controlnet to be available.
        """
        # Freeze inverse LoRA: only back_blocks update from FM loss
        feat = self.accelerator.unwrap_model(self.model).feature_extractor
        feat.freeze_lora()

        if self.forward_controlnet is None:
            feat.unfreeze_lora()
            return torch.tensor(0.0, device=self.accelerator.device), {
                "loss_fm_forward": 0.0, "loss_total": 0.0, "step_type": 3.0,
            }

        device = self.accelerator.device
        dtype  = torch.bfloat16

        hair      = batch["img"].to(device, dtype=dtype)
        sketch_gt = batch["sketch"].to(device, dtype=dtype)
        matte_gt  = batch["matte"].to(device, dtype=dtype)
        B = hair.shape[0]

        if update_optimizer:
            self.optimizer.zero_grad()

        with torch.no_grad():
            hair_latent = self.vae.encode(hair)   # (B, 16, 64, 64)

        # Rectified flow: x_t = (1-t)*x_0 + t*noise,  v_target = noise - x_0
        t     = torch.rand(B, device=device, dtype=dtype)
        noise = torch.randn_like(hair_latent)
        x_t   = (1 - t.view(B, 1, 1, 1)) * hair_latent + t.view(B, 1, 1, 1) * noise
        v_target = (noise - hair_latent).detach()

        # ControlNet conditioning — trainable(Option B): grad 허용, frozen: no_grad
        if self.controlnet_trainable:
            block_samples = self.forward_controlnet._get_features_impl(
                sketch_gt.float(), matte_gt.float(), enable_grad=True,
            )
        else:
            with torch.no_grad():
                block_samples = self.forward_controlnet._get_features_impl(
                    sketch_gt.float(), matte_gt.float(), enable_grad=False,
                )

        null_enc = self._null_enc_hs.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_p   = self._null_pooled.expand(B, -1).to(device=device, dtype=dtype)

        # hook + checkpoint(use_reentrant=False) 조합은 backward 재연산 시 hook이 없어
        # CheckpointError를 유발한다. block_controlnet_hidden_states 네이티브 파라미터를
        # 사용하고 block_samples를 checkpoint 입력으로 직접 전달해 회피한다.
        def _tx_fwd(latent, enc, pool, t_, *blocks):
            return self.transformer(
                hidden_states=latent,
                encoder_hidden_states=enc,
                pooled_projections=pool,
                timestep=t_,
                block_controlnet_hidden_states=list(blocks),
                return_dict=False,
            )[0]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = checkpoint(
                _tx_fwd, x_t, null_enc, null_p, t, *block_samples,
                use_reentrant=False,
            )

        fm_loss = F.mse_loss(v_pred.float(), v_target.float()) * self.w_fm_forward
        self.accelerator.backward(fm_loss)

        if update_optimizer:
            if self.cfg["training"].get("gradient_clip"):
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg["training"]["gradient_clip"],
                )
            self.optimizer.step()

        # Restore inverse LoRA gradients for next supervised step
        feat.unfreeze_lora()

        log = {
            "loss_fm_forward": fm_loss.item() / max(self.w_fm_forward, 1e-8),
            "loss_total":      fm_loss.item(),
            "step_type":       3.0,
        }
        return fm_loss, log

    def _compute_image_lpips_cycle(
        self,
        sketch_pred: torch.Tensor,   # (B, 3, 512, 512) float32, gradient-tracked
        matte_gt:    torch.Tensor,   # (B, 1, 512, 512) float32 in [0, 1]
        hair_image:  torch.Tensor,   # (B, 3, 512, 512) float32 in [0, 1]
    ) -> torch.Tensor:
        """
        Round-trip cycle:

            sketch_pred + matte_gt → forward_controlnet → block_samples
            sketch_pred → vae.encode_for_grad → sketch_latent
            DiT(sketch_latent, σ=1, null_cond, block_samples injected)
              → v_pred → pred_hair_latent = sketch_latent - v_pred
            vae.decode(pred_hair_latent) → hair_recon
            loss = LPIPS(hair_recon * matte_gt, hair_orig * matte_gt)

        Gradient chain:
          LPIPS → hair_recon → vae.decode → pred_hair_latent
                → DiT (back_blocks 12-23 receive grads)
                → sketch_latent → vae.encode_for_grad → sketch_pred
                                                       → stroke_decoder ✓
                                                       → feature_extractor ✓

        autocast(bf16) required: transformer called directly, not via accelerator-wrapped model.
        """
        B      = sketch_pred.shape[0]
        device = sketch_pred.device
        dtype  = sketch_pred.dtype

        # 1. ControlNet residuals: sketch_pred + matte_gt → block_samples
        block_samples = self.forward_controlnet._get_features_impl(
            sketch_pred, matte_gt, enable_grad=True,
        )

        # 2. DiT input: sketch_latent (the "noisy" thing the DiT will denoise toward hair)
        sketch_latent = self.vae.encode_for_grad(sketch_pred).to(dtype=torch.bfloat16)

        # 3. DiT forward with ControlNet injection via hooks
        null_enc = self._null_enc_hs.expand(B, -1, -1).to(device=device, dtype=torch.bfloat16)
        null_p   = self._null_pooled.expand(B, -1).to(device=device, dtype=torch.bfloat16)
        
        # 1-Step Fix: sigma=1.0 (noise state) instead of 0.0 (image state)
        sigma    = torch.ones(B, device=device, dtype=torch.bfloat16) * 1.0

        # Hook factory: closure captures `sample` correctly per iteration
        def make_inject_hook(sample):
            def _h(_module, _inputs, outputs):
                # SD3 JointTransformerBlock returns (encoder_hs, image_hs)
                enc_hs, img_hs = outputs
                s = sample.to(dtype=img_hs.dtype)
                # Match seq_len: img_hs may have more tokens than image-only sample
                n = s.shape[1]
                if img_hs.shape[1] == n:
                    new_img_hs = img_hs + s
                else:
                    head = img_hs[:, :n, :] + s
                    new_img_hs = torch.cat([head, img_hs[:, n:, :]], dim=1)
                return (enc_hs, new_img_hs)
            return _h

        n_inject = min(len(block_samples), len(self.transformer.transformer_blocks))
        hooks = []
        try:
            for i in range(n_inject):
                hk = self.transformer.transformer_blocks[i].register_forward_hook(
                    make_inject_hook(block_samples[i])
                )
                hooks.append(hk)

            # Gradient checkpointing on this transformer pass (~10GB memory saved).
            def _tx_fwd(latent, enc, pool, t):
                return self.transformer(
                    hidden_states=latent,
                    encoder_hidden_states=enc,
                    pooled_projections=pool,
                    timestep=t,
                    return_dict=False,
                )[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = checkpoint(
                    _tx_fwd, sketch_latent, null_enc, null_p, sigma,
                    use_reentrant=False,
                )   # (B, 16, 64, 64)
                
                # 1-Step Fix: x_0 = x_1 - v_pred (SD3 Flow Matching Euler step)
                pred_hair_latent = sketch_latent - v_pred
        finally:
            for hk in hooks:
                hk.remove()

        # 4. Decode → image space in [0, 1]
        hair_recon_01 = (self.vae.decode(pred_hair_latent).to(dtype=dtype).clamp(-1.0, 1.0) + 1.0) / 2.0

        # Mask both in [0, 1] space using matte_gt (prevents trivial shortcut)
        recon_masked = hair_recon_01 * matte_gt.to(dtype=dtype)
        ref_masked   = hair_image.to(dtype=dtype) * matte_gt.to(dtype=dtype)

        # Convert back to [-1, 1] for LPIPS
        recon_11 = recon_masked * 2.0 - 1.0
        ref_11   = ref_masked * 2.0 - 1.0

        # 5. LPIPS at 256×256 + bf16 — VGG memory ~8× cheaper than 512×512 fp32
        recon_lp = F.interpolate(recon_11, size=(256, 256), mode="bilinear", align_corners=False)
        ref_lp   = F.interpolate(ref_11, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lpips_val = self.lpips_fn(recon_lp, ref_lp).mean()
        return lpips_val.float()

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        device = self.accelerator.device
        dtype  = torch.bfloat16
        total, n = 0.0, 0

        for batch in self.val_loader:
            hair_image = batch["img"].to(device, dtype=dtype)
            sketch_gt  = batch["sketch"].to(device, dtype=dtype)
            matte_gt   = batch["matte"].to(device, dtype=dtype)

            sketch_pred, stroke_mask = self.model(hair_image, matte_gt)
            _, log = self.loss_fn(stroke_mask, sketch_pred, sketch_gt)
            total += log["loss_total"]
            n += 1

        self.model.train()
        return total / max(n, 1)

    @torch.no_grad()
    def _validate_forward(self) -> float:
        """Forward regression monitor: ControlNet FM loss on val set."""
        if self.forward_controlnet is None:
            return 0.0
        cn = self.accelerator.unwrap_model(self.forward_controlnet)
        cn.eval()
        device, dtype = self.accelerator.device, torch.bfloat16
        total, n = 0.0, 0

        for batch in self.val_loader:
            sketch_gt = batch["sketch"].to(device, dtype=dtype)
            matte_gt  = batch["matte"].to(device, dtype=dtype)
            hair      = batch["img"].to(device, dtype=dtype)
            B = hair.shape[0]

            hair_latent = self.vae.encode(hair)
            t       = torch.rand(B, device=device, dtype=dtype)
            noise   = torch.randn_like(hair_latent)
            x_t     = (1 - t.view(B, 1, 1, 1)) * hair_latent + t.view(B, 1, 1, 1) * noise
            v_target = (noise - hair_latent).detach()

            block_samples, null_enc_hs, null_pooled = cn(
                noisy_latent=x_t,
                sketch=sketch_gt.float(),
                matte=matte_gt.float(),
                sigmas=t,
            )
            v_pred = self.transformer(
                hidden_states=x_t,
                encoder_hidden_states=null_enc_hs.to(dtype=dtype),
                pooled_projections=null_pooled.to(dtype=dtype),
                timestep=t,
                block_controlnet_hidden_states=[s.to(dtype=dtype) for s in block_samples],
                return_dict=False,
            )[0]
            total += F.mse_loss(v_pred.float(), v_target.float()).item()
            n += 1

        if self.controlnet_trainable:
            self.accelerator.unwrap_model(self.forward_controlnet).train()
        return total / max(n, 1)

    @torch.no_grad()
    def _log_images(self, epoch: int, n_samples: int = 4):
        if not self.accelerator.is_main_process:
            return
        try:
            import numpy as np
        except ImportError:
            return

        # tensorboard writer (single backend — no wandb)
        tb_writer = None
        try:
            tb = self.accelerator.get_tracker("tensorboard", unwrap=True)
            tb_writer = tb if hasattr(tb, "add_image") else getattr(tb, "writer", None)
        except Exception:
            tb_writer = None
        if tb_writer is None:
            return

        device = self.accelerator.device
        dtype  = torch.bfloat16
        batch  = next(iter(self.val_loader))

        hair_image = batch["img"][:n_samples].to(device, dtype=dtype)
        sketch_gt  = batch["sketch"][:n_samples]
        matte_gt   = batch["matte"][:n_samples]

        self.model.eval()
        sketch_pred, _ = self.model(hair_image, matte_gt.to(device, dtype=dtype))

        # Phase 2: compute hair_recon for cycle visualization (same path as training cycle)
        log_recon = (epoch >= self.cycle_start) and (self.forward_controlnet is not None)
        if log_recon:
            B = hair_image.shape[0]
            block_samples = self.forward_controlnet._get_features_impl(
                sketch_pred.float(), matte_gt.to(device, dtype=torch.float32), enable_grad=False,
            )
            sketch_latent = self.vae.encode(sketch_pred.float()).to(dtype=torch.bfloat16)
            null_enc = self._null_enc_hs.expand(B, -1, -1).to(device=device, dtype=torch.bfloat16)
            null_p   = self._null_pooled.expand(B, -1).to(device=device, dtype=torch.bfloat16)
            sigma    = torch.zeros(B, device=device, dtype=torch.bfloat16)

            # Inject block_samples via hooks (same as training cycle)
            def _make_hook(sample):
                def _h(_m, _i, outputs):
                    enc_hs, img_hs = outputs
                    s = sample.to(dtype=img_hs.dtype)
                    n = s.shape[1]
                    if img_hs.shape[1] == n:
                        new_img = img_hs + s
                    else:
                        new_img = torch.cat([img_hs[:, :n] + s, img_hs[:, n:]], dim=1)
                    return (enc_hs, new_img)
                return _h

            n_inject = min(len(block_samples), len(self.transformer.transformer_blocks))
            hooks = []
            try:
                for i in range(n_inject):
                    hooks.append(
                        self.transformer.transformer_blocks[i].register_forward_hook(
                            _make_hook(block_samples[i])
                        )
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_lat = self.transformer(
                        hidden_states=sketch_latent,
                        encoder_hidden_states=null_enc,
                        pooled_projections=null_p,
                        timestep=sigma,
                        return_dict=False,
                    )[0]
            finally:
                for hk in hooks:
                    hk.remove()

            hair_recon = self.vae.decode(pred_lat)               # [-1, 1]
            hair_recon = VAEWrapper.denormalize(hair_recon).clamp(0, 1)

        def to_np(t):
            return (t.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

        cap = "hair | sketch_pred | sketch_GT"
        if log_recon:
            cap += " | hair_recon (cycle)"

        panel_arrays = []
        for i in range(min(n_samples, hair_image.shape[0])):
            cols = [
                to_np(hair_image)[i],
                to_np(sketch_pred)[i],
                to_np(sketch_gt)[i],
            ]
            if log_recon:
                cols.append(to_np(hair_recon)[i])
            panel_arrays.append(np.concatenate(cols, axis=1))

        # tensorboard image logging (only backend)
        for i, p in enumerate(panel_arrays):
            tb_writer.add_image(
                f"val/sample_{i}", p.transpose(2, 0, 1),
                self.global_step, dataformats="CHW",
            )

        # --- Block importance: log as scalar dict (works in both backends) ---
        # Each block's weight per scale → flat metric so it shows in any tracker.
        unwrapped = self.accelerator.unwrap_model(self.model)
        block_imp = unwrapped.feature_extractor.get_block_importance()
        flat = {
            f"block_importance/{scale}/blk{i:02d}": float(w[i])
            for scale, w in block_imp.items()
            for i in range(w.shape[0])
        }
        self.accelerator.log(flat, step=self.global_step)

    def _save(self, filename: str, epoch: int):
        if not self.accelerator.is_main_process:
            return
        ckpt = {
            "model":         self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "lr_scheduler":  self.lr_scheduler.state_dict(),
            "global_step":   self.global_step,
            "epoch":         epoch + 1,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(ckpt, self.output_dir / filename)
        self.accelerator.print(f"Saved: {self.output_dir / filename}")

        # Option B: save ControlNet separately for regression rollback
        if self.controlnet_trainable and self.forward_controlnet is not None:
            cn_ckpt = {
                "controlnet": self.accelerator.unwrap_model(self.forward_controlnet).state_dict(),
                "optimizer_cn":    self.optimizer_cn.state_dict(),
                "lr_scheduler_cn": self.lr_scheduler_cn.state_dict(),
                "epoch":           epoch + 1,
            }
            cn_filename = filename.replace(".pth", "_controlnet.pth")
            torch.save(cn_ckpt, self.output_dir / cn_filename)
            self.accelerator.print(f"Saved: {self.output_dir / cn_filename}")


def _flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float, str, bool)):
            out[key] = v
        else:
            out[key] = str(v)
    return out
