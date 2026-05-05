"""
InverseHeadTrainer: Hair→Sketch via partially-unfrozen DiT + trainable FPN mask decoder.

Backbone:
  blocks 0-11:  frozen base + LoRA (to_q/k/v, rank-8)
  blocks 12-23: fully unfrozen (all parameters trainable, lower LR via backbone_lr_scale)

Loss schedule:
  Phase 1 (epoch < cycle_start):
    Every step → supervised (structure BCE + color L1 + matte BCE+Dice+L1 + TV)

  Phase 2 (epoch >= cycle_start):
    Even global_step → supervised step (same as phase 1)
    Odd  global_step → cycle step:
        - feature MSE: MSE(forward_cn.features(sketch_pred), forward_cn.features(sketch_gt))
        - image LPIPS: LPIPS(vae.decode(dit(encode_for_grad(sketch_pred))), hair_orig)

Mini-batch alternating design:
  Two separate zero_grad/backward/step pairs per alternation period.
  The feature cycle and LPIPS cycle both use enable_grad=True paths so that
  gradients reach the inverse model parameters through frozen forward models.
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
        self.w_cycle     = tcfg["loss_weights"].get("cycle", 0.01)
        self.w_lpips     = tcfg["loss_weights"].get("lpips_cycle", 0.05)
        self.cycle_start = tcfg.get("cycle_start", 9999)
        if (self.w_cycle > 0 or self.w_lpips > 0) and Path(controlnet_ckpt).exists():
            fwd_cn = HairControlNet(
                model_id=model_id, vae=self.vae,
                num_layers=config["model"].get("num_controlnet_layers", 12),
                local_files_only=local_only,
            )
            ckpt = torch.load(controlnet_ckpt, map_location="cpu", weights_only=True)
            fwd_cn.load_state_dict(ckpt["controlnet"])
            fwd_cn.eval()
            for p in fwd_cn.parameters():
                p.requires_grad_(False)
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

        # --- Loss ---
        lw = tcfg.get("loss_weights", {})
        self.loss_fn = InversionLoss(
            w_structure=lw.get("structure", 1.0),
            w_color=lw.get("color", 0.5),
            w_matte=lw.get("matte", 1.0),
        )
        self.w_tv = lw.get("tv", 0.01)

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Accelerate prepare ---
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer,
            self.train_loader, self.val_loader,
            self.lr_scheduler,
        )
        device = self.accelerator.device
        self.vae         = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        if self.forward_controlnet is not None:
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

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.start_epoch   = 0
        self._current_epoch = 0
        self._restore_training_state()

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

    def train(self):
        tcfg       = self.cfg["training"]
        epochs     = tcfg.get("epochs", 50)
        save_every = self.cfg["checkpointing"].get("save_every", 10)
        eval_every = self.cfg["checkpointing"].get("eval_every", 5)

        for epoch in range(self.start_epoch, epochs):
            self._current_epoch = epoch
            phase2 = (epoch >= self.cycle_start)

            self.model.train()
            epoch_losses = []

            desc = f"Epoch {epoch+1}/{epochs}" + (" [+cycle-alt]" if phase2 else "")
            progress = tqdm(
                self.train_loader, desc=desc,
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress:
                # Phase 1: supervised only.
                # Phase 2: alternate — even steps supervised, odd steps cycle.
                if phase2 and self.global_step % 2 == 1:
                    loss, log_dict = self._cycle_step(batch)
                else:
                    loss, log_dict = self._supervised_step(batch)

                epoch_losses.append(log_dict["loss_total"])

                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        # Each group has its own _base_lr (main vs back_blocks at lower LR).
                        # CosineAnnealingLR will continue from these once warmup ends.
                        pg["lr"] = pg["_base_lr"] * lr_scale
                else:
                    self.lr_scheduler.step()

                self.global_step += 1
                progress.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
                self.accelerator.log(log_dict, step=self.global_step)

            avg = sum(epoch_losses) / len(epoch_losses)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

            if (epoch + 1) % eval_every == 0:
                val_loss = self._validate()
                self.accelerator.print(f"Val loss: {val_loss:.4f}")
                self.accelerator.log({"val_loss": val_loss, "epoch": epoch + 1}, step=self.global_step)

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

    def _supervised_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Supervised loss only: structure BCE + color L1 + matte losses + TV."""
        device = self.accelerator.device
        dtype  = torch.bfloat16

        hair_image = batch["img"].to(device, dtype=dtype)
        sketch_gt  = batch["sketch"].to(device, dtype=dtype)
        matte_gt   = batch["matte"].to(device, dtype=dtype)

        self.optimizer.zero_grad()

        sketch_pred, matte_pred, stroke_mask = self.model(hair_image)

        loss, log_dict = self.loss_fn(
            stroke_mask_pred=stroke_mask,
            sketch_pred=sketch_pred,
            matte_pred=matte_pred,
            sketch_gt=sketch_gt,
            matte_gt=matte_gt,
        )

        l_tv = tv_loss(stroke_mask.float())
        loss = loss + self.w_tv * l_tv
        log_dict["loss_tv"]    = l_tv.item()
        log_dict["loss_total"] = loss.item()   # refresh after TV term
        log_dict["step_type"]  = 0.0           # supervised=0 for logging

        self.accelerator.backward(loss)
        if self.cfg["training"].get("gradient_clip"):
            self.accelerator.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg["training"]["gradient_clip"],
            )
        self.optimizer.step()
        return loss, log_dict

    def _cycle_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Cycle loss step (phase 2): feature MSE + image-space LPIPS cycle.

        Both losses use enable_grad=True paths so that the gradient propagates
        back through frozen forward models to the inverse model parameters.
        """
        device = self.accelerator.device
        dtype  = torch.bfloat16

        hair_image = batch["img"].to(device, dtype=dtype)
        sketch_gt  = batch["sketch"].to(device, dtype=dtype)
        matte_gt   = batch["matte"].to(device, dtype=dtype)

        # If both cycle losses disabled, fall back to supervised step
        # (this can happen if forward_controlnet failed to load and lpips is missing).
        feat_active  = self.forward_controlnet is not None and self.w_cycle > 0
        lpips_active = self.w_lpips > 0 and self.lpips_fn is not None
        if not feat_active and not lpips_active:
            return self._supervised_step(batch)

        self.optimizer.zero_grad()

        # Fresh inverse model forward (with full gradient tracking)
        sketch_pred, matte_pred, stroke_mask = self.model(hair_image)

        loss_terms: list[torch.Tensor] = []
        log_dict: dict = {}

        # --- Feature cycle: MSE between ControlNet features of pred vs GT ---
        if feat_active:
            # enable_grad=True: gradient flows through frozen ControlNet → sketch_pred → inverse model
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
            log_dict["loss_cycle_feat"] = feat_cycle.item()

        # --- Image-space LPIPS cycle: LPIPS(vae.decode(dit(encode(sketch_pred))), hair_orig) ---
        if lpips_active:
            lpips_loss = self._compute_image_lpips_cycle(sketch_pred.float(), hair_image.float())
            loss_terms.append(self.w_lpips * lpips_loss)
            log_dict["loss_cycle_lpips"] = lpips_loss.item()

        # Sum loss terms — guaranteed non-empty by the early-return above
        total_cycle = sum(loss_terms[1:], loss_terms[0])

        log_dict["loss_total"] = total_cycle.item()
        log_dict["step_type"]  = 1.0  # cycle=1 for logging

        self.accelerator.backward(total_cycle)
        if self.cfg["training"].get("gradient_clip"):
            self.accelerator.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg["training"]["gradient_clip"],
            )
        self.optimizer.step()
        return total_cycle, log_dict

    def _compute_image_lpips_cycle(
        self,
        sketch_pred: torch.Tensor,   # (B, 3, 512, 512) float32, gradient-tracked
        hair_image:  torch.Tensor,   # (B, 3, 512, 512) float32 in [0, 1], no grad needed
    ) -> torch.Tensor:
        """
        sketch_pred → vae.encode_for_grad → sketch_latent
                    → transformer(sigma=0, null_cond) → pred_hair_latent
                    → vae.decode → hair_recon (in [-1, 1])
        LPIPS(hair_recon, normalize(hair_image))

        The VAE and transformer are frozen (requires_grad=False for their params),
        but gradients DO flow through them back to sketch_pred because:
          - vae.encode_for_grad has no torch.no_grad() wrapper
          - vae.decode has no torch.no_grad() wrapper
          - transformer back_blocks (12-23) are unfrozen and also receive grads here

        NOTE (multi-GPU): self.transformer is NOT wrapped by accelerator.prepare(),
        so back_blocks gradients accumulated through this call bypass DDP sync.
        Tested with single-GPU; for multi-GPU use, route through self.model or
        wrap self.transformer separately.
        """
        B      = sketch_pred.shape[0]
        device = sketch_pred.device
        dtype  = sketch_pred.dtype

        # sketch_pred [0,1] → sketch_latent (with grad)
        sketch_latent = self.vae.encode_for_grad(sketch_pred)   # (B, 16, 64, 64)
        sketch_latent = sketch_latent.to(dtype=torch.bfloat16)

        # Run DiT (sigma=0) on sketch_latent to predict hair-like latent.
        # No ControlNet injection here for simplicity — bare DiT single-step prediction.
        # Back_blocks (12-23) are unfrozen and also get LPIPS gradients.
        null_enc = self._null_enc_hs.expand(B, -1, -1).to(device=device, dtype=torch.bfloat16)
        null_p   = self._null_pooled.expand(B, -1).to(device=device, dtype=torch.bfloat16)
        sigma    = torch.zeros(B, device=device, dtype=torch.bfloat16)

        # Gradient checkpointing on this transformer pass (single biggest activation
        # source in cycle step). Forward saves no activations; backward recomputes.
        # Trades ~30% extra compute for ~10GB memory — keeps batch_size=4 viable.
        # use_reentrant=False: PyTorch >=2.0 API, hook-safe (no double firing).
        def _tx_fwd(latent, enc, pool, t):
            return self.transformer(
                hidden_states=latent,
                encoder_hidden_states=enc,
                pooled_projections=pool,
                timestep=t,
                return_dict=False,
            )[0]
        pred_hair_latent = checkpoint(
            _tx_fwd, sketch_latent, null_enc, null_p, sigma,
            use_reentrant=False,
        )   # (B, 16, 64, 64)

        # Decode to pixel space (VAE params frozen, but grad flows to input)
        hair_recon = self.vae.decode(pred_hair_latent).to(dtype=dtype)  # (B, 3, 512, 512) in [-1,1]

        # hair_image is [0,1]; normalize to [-1,1] for LPIPS comparison
        hair_ref = VAEWrapper.normalize(hair_image)   # (B, 3, 512, 512) in [-1,1]

        return self.lpips_fn(
            hair_recon.clamp(-1.0, 1.0),
            hair_ref.clamp(-1.0, 1.0),
        ).mean()

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

            sketch_pred, matte_pred, stroke_mask = self.model(hair_image)
            _, log = self.loss_fn(stroke_mask, sketch_pred, matte_pred, sketch_gt, matte_gt)
            total += log["loss_total"]
            n += 1

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
        sketch_pred, matte_pred, _ = self.model(hair_image)

        # Phase 2: also compute hair_recon for cycle visualization
        log_recon = (epoch >= self.cycle_start) and (self.lpips_fn is not None)
        if log_recon:
            sketch_latent = self.vae.encode_for_grad(sketch_pred.float()).to(dtype=torch.bfloat16)
            B = sketch_latent.shape[0]
            null_enc = self._null_enc_hs.expand(B, -1, -1).to(device=device, dtype=torch.bfloat16)
            null_p   = self._null_pooled.expand(B, -1).to(device=device, dtype=torch.bfloat16)
            sigma    = torch.zeros(B, device=device, dtype=torch.bfloat16)
            pred_lat = self.transformer(
                hidden_states=sketch_latent,
                encoder_hidden_states=null_enc,
                pooled_projections=null_p,
                timestep=sigma,
                return_dict=False,
            )[0]
            hair_recon = self.vae.decode(pred_lat)               # [-1, 1]
            hair_recon = VAEWrapper.denormalize(hair_recon).clamp(0, 1)

        def to_np(t):
            return (t.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

        cap = "hair | sketch_pred | sketch_GT | matte_pred | matte_GT"
        if log_recon:
            cap += " | hair_recon (cycle)"

        # Build panels as raw numpy arrays (HWC uint8)
        panel_arrays = []
        for i in range(min(n_samples, hair_image.shape[0])):
            cols = [
                to_np(hair_image)[i],
                to_np(sketch_pred)[i],
                to_np(sketch_gt)[i],
                to_np(matte_pred.repeat(1, 3, 1, 1))[i],
                to_np(matte_gt.repeat(1, 3, 1, 1))[i],
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
