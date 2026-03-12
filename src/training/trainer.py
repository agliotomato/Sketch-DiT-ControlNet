"""
Trainer for SD3.5 ControlNet hair region generation.

Holds:
  - HairControlNet (trainable): ControlNet + MatteCNN + learned null embeddings
  - SD3Transformer2DModel (frozen): generates velocity predictions via ControlNet residuals
  - VAEWrapper SD3.5 (frozen): 16-channel latents
  - FlowMatchEulerDiscreteScheduler: flow matching noise schedule

Phase 1 (pretrain):
  - Dataset: unbraid (3K samples), 200 epochs
  - Only HairControlNet.parameters() trained
  - Transformer fully frozen

Phase 2 (finetune):
  - Dataset: braid (1K samples), 100 epochs
  - Load Phase 1 checkpoint
  - HairControlNet trained with lower lr
  - Transformer still frozen

Training step:
  1. target → VAE encode → latents (16ch, 64x64)
  2. Sample sigma from logit-normal distribution
  3. noisy_latents = (1-sigma)*latents + sigma*noise
  4. HairControlNet(noisy_latents, sketch, matte, sigmas) → block_samples, null_hs, null_pooled
  5. transformer(noisy_latents, null_hs, null_pooled, sigmas, block_samples) → v_pred
  6. v_target = noise - latents
  7. loss = flow_loss + lpips + edge
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairRegionDataset
from src.data.utils import resize_matte_to_latent
from src.models.controlnet_sd35 import HairControlNet
from src.models.vae_wrapper import VAEWrapper
from src.training.ema import EMAModel
from src.training.losses import HairLoss


class Trainer:
    """
    Unified trainer for Phase 1 (pretrain) and Phase 2 (finetune).

    Args:
        config: dict loaded from YAML config file
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.phase = config["training"]["phase"]
        assert self.phase in ("pretrain", "finetune"), f"Unknown phase: {self.phase}"

        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 2),
            log_with="tensorboard",
            project_dir=config["checkpointing"]["output_dir"],
        )

        self._setup_models()
        self._setup_data()
        self._setup_optimizer()
        self._prepare_accelerator()

        self.loss_fn = HairLoss(
            phase=self.phase,
            w_flow=config["training"]["loss_weights"].get("flow", 1.0),
            w_lpips=config["training"]["loss_weights"].get("lpips", 0.1),
            w_edge=config["training"]["loss_weights"].get("edge", 0.0),
            lpips_warmup_frac=config["training"]["loss_weights"].get("lpips_warmup_frac", 0.3),
        )

        self.ema = EMAModel(
            self.accelerator.unwrap_model(self.controlnet),
            decay=config["training"].get("ema_decay", 0.9999),
        )

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Flow matching timestep sampling hyperparams
        self.logit_mean = config["training"].get("logit_mean", 0.0)
        self.logit_std  = config["training"].get("logit_std",  1.0)

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_models(self):
        cfg = self.cfg
        model_id = cfg["model"]["model_id"]
        local_files_only = cfg.get("local_files_only", False)

        # Frozen SD3.5 VAE (bfloat16)
        self.vae = VAEWrapper.from_pretrained(
            model_id=model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        self.vae.eval()

        # Frozen SD3.5 Transformer
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.transformer.eval()

        # Trainable HairControlNet (builds ControlNet from transformer weights)
        num_layers = cfg["model"].get("num_controlnet_layers", 12)
        self.controlnet = HairControlNet(
            model_id=model_id,
            vae=self.vae,
            num_layers=num_layers,
            local_files_only=local_files_only,
        )

        # Gradient checkpointing: saves ~40% activation memory at ~20% compute cost
        # Required when LPIPS is active: VAE decode computation graph adds significant memory
        if cfg["training"].get("gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()
            self.controlnet.controlnet.enable_gradient_checkpointing()
            self.vae.vae.enable_gradient_checkpointing()

        # Flow matching scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Load Phase 1 checkpoint for Phase 2
        resume_from = cfg["training"].get("resume_from")
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location="cpu")
            self.controlnet.load_state_dict(ckpt["controlnet"])
            self.accelerator.print(f"Loaded checkpoint from {resume_from}")

    def _setup_data(self):
        cfg = self.cfg["training"]
        aug = build_augmentation_pipeline(self.phase)

        split_train = f"{cfg['dataset']}_train"
        split_val   = f"{cfg['dataset']}_test"

        train_ds = HairRegionDataset(split=split_train, augmentation=aug)
        val_ds   = HairRegionDataset(split=split_val)

        bs = cfg.get("batch_size", 4)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
        )

    def _setup_optimizer(self):
        cfg = self.cfg["training"]
        lr = cfg.get("learning_rate", 1e-4)

        self.optimizer = AdamW(
            self.controlnet.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        epochs = cfg.get("epochs", 200)
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        warmup = cfg.get("warmup_steps", 500)

        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup, 1),
            eta_min=1e-6,
        )
        self.warmup_steps = warmup
        self.total_steps = total_steps

    def _prepare_accelerator(self):
        (
            self.controlnet,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.controlnet,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        )
        # VAE and transformer stay on device but are not managed by Accelerator
        device = self.accelerator.device
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)

    # ------------------------------------------------------------------
    # Timestep / sigma sampling (logit-normal for flow matching)
    # ------------------------------------------------------------------

    def _sample_sigmas(self, bsz: int, device: torch.device) -> torch.Tensor:
        """
        Sample sigma values using logit-normal distribution.

        Returns:
            sigmas: (B, 1, 1, 1) suitable for broadcasting against (B, 16, 64, 64)
        """
        n_train = self.scheduler.config.num_train_timesteps
        u = torch.normal(
            mean=self.logit_mean,
            std=self.logit_std,
            size=(bsz,),
            device=device,
        )
        u = torch.sigmoid(u)  # [0, 1]
        # Map to scheduler sigma values
        indices = (u * n_train).long().clamp(0, n_train - 1)
        sigmas = self.scheduler.sigmas[indices.cpu()].to(device=device)  # (B,)
        return sigmas.view(bsz, 1, 1, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg["training"]
        epochs = cfg.get("epochs", 200)
        eval_every = self.cfg["checkpointing"].get("eval_every", 10)
        save_every = self.cfg["checkpointing"].get("save_every", 20)
        grad_clip = cfg.get("gradient_clip", 1.0)

        self.accelerator.print(f"Starting {self.phase} training for {epochs} epochs")

        for epoch in range(epochs):
            self.controlnet.train()

            epoch_losses = []
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress:
                loss, log_dict = self._train_step(batch, grad_clip=grad_clip)
                epoch_losses.append(log_dict["loss_total"])

                # Linear warmup then cosine decay
                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.cfg["training"]["learning_rate"] * lr_scale
                else:
                    self.lr_scheduler.step()

                # EMA update on unwrapped controlnet
                self.ema.update(self.accelerator.unwrap_model(self.controlnet))
                self.global_step += 1

                progress.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
                self.accelerator.log(log_dict, step=self.global_step)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            if (epoch + 1) % eval_every == 0:
                val_loss = self._validate()
                self.accelerator.print(f"Val loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pth")

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth")

        self._save_checkpoint("final.pth")
        self.accelerator.end_training()

    def _train_step(self, batch: dict, grad_clip: float = 1.0) -> tuple[torch.Tensor, dict]:
        sketch = batch["sketch"]   # (B, 3, 512, 512) [0,1]
        matte  = batch["matte"]    # (B, 1, 512, 512) [0,1]
        target = batch["target"]   # (B, 3, 512, 512) [0,1]

        with self.accelerator.accumulate(self.controlnet):
            device = self.accelerator.device
            B = target.shape[0]

            # 1. Encode target image → latents
            with torch.no_grad():
                latents = self.vae.encode(target)  # (B, 16, 64, 64)

            # 2. Sample sigma (flow matching)
            sigmas = self._sample_sigmas(B, device)  # (B, 1, 1, 1)

            # 3. Sample noise and form noisy latent
            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # 4. Matte at latent resolution for loss weighting
            matte_latent = resize_matte_to_latent(matte)  # (B, 1, 64, 64)

            # Cast inputs to bfloat16 for model forward passes
            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            sigmas_1d = sigmas.view(B).to(dtype=torch.bfloat16)  # (B,) transformer expects bf16

            # 5. ControlNet forward → block_samples + null conditioning
            block_samples, null_enc_hs, null_pooled = self.controlnet(
                noisy_latent=noisy_latents,
                sketch=sketch,
                matte=matte,
                sigmas=sigmas_1d,
            )
            # Accelerator's convert_to_fp32 converts controlnet outputs to fp32.
            # Cast back to bf16 so the frozen bf16 transformer can consume them.
            block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
            null_enc_hs   = null_enc_hs.to(dtype=torch.bfloat16)
            null_pooled   = null_pooled.to(dtype=torch.bfloat16)

            # 6. Frozen transformer forward with ControlNet residuals
            # NOTE: do NOT use torch.no_grad() here.
            # Transformer parameters are frozen via requires_grad_(False),
            # but the computation graph must be maintained so that gradients
            # flow back through block_controlnet_hidden_states → HairControlNet.
            v_pred = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=null_enc_hs,
                pooled_projections=null_pooled,
                timestep=sigmas_1d,
                block_controlnet_hidden_states=block_samples,
                return_dict=False,
            )[0]   # (B, 16, 64, 64)

            # 7. Flow matching velocity target: v = noise - latents
            v_target = (noise - latents).to(dtype=torch.bfloat16)

            # 8. Loss
            total_loss, log_dict = self.loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                matte_latent=matte_latent.to(dtype=torch.bfloat16),
                x_t=noisy_latents,
                sigmas=sigmas.to(dtype=torch.bfloat16),
                vae=self.vae,
                target_rgb=target,
                sketch=sketch,
                matte=matte,
                current_step=self.global_step,
                total_steps=self.total_steps,
            )

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.controlnet.parameters(),
                    max_norm=grad_clip,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss, log_dict

    @torch.no_grad()
    def _validate(self) -> float:
        self.controlnet.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            sketch = batch["sketch"]
            matte  = batch["matte"]
            target = batch["target"]

            device = self.accelerator.device
            B = target.shape[0]

            latents = self.vae.encode(target)
            sigmas = self._sample_sigmas(B, device)
            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            matte_latent = resize_matte_to_latent(matte)

            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            sigmas_1d = sigmas.view(B).to(dtype=torch.bfloat16)

            block_samples, null_enc_hs, null_pooled = self.controlnet(
                noisy_latent=noisy_latents,
                sketch=sketch,
                matte=matte,
                sigmas=sigmas_1d,
            )
            block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
            null_enc_hs   = null_enc_hs.to(dtype=torch.bfloat16)
            null_pooled   = null_pooled.to(dtype=torch.bfloat16)

            v_pred = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=null_enc_hs,
                pooled_projections=null_pooled,
                timestep=sigmas_1d,
                block_controlnet_hidden_states=block_samples,
                return_dict=False,
            )[0]

            v_target = (noise - latents).to(dtype=torch.bfloat16)

            _, log_dict = self.loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                matte_latent=matte_latent.to(dtype=torch.bfloat16),
            )
            total_loss += log_dict["loss_total"]
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str):
        if not self.accelerator.is_main_process:
            return
        controlnet_unwrapped = self.accelerator.unwrap_model(self.controlnet)
        ckpt = {
            "controlnet":  controlnet_unwrapped.state_dict(),
            "ema":         self.ema.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config":      self.cfg,
        }
        save_path = self.output_dir / filename
        torch.save(ckpt, save_path)
        self.accelerator.print(f"Saved checkpoint: {save_path}")
