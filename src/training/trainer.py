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
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairRegionDataset
from src.data.utils import resize_matte_to_latent
from src.models.controlnet_sd35 import HairControlNet, gate_block_samples
from src.models.vae_wrapper import VAEWrapper
from src.training.ema import EMAModel
from src.training.losses import HairLoss


def _flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Nested dict → flat dict with dot-separated keys (TensorBoard hparam 호환)."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, key))
        elif isinstance(v, (int, float, str, bool)):
            out[key] = v
        elif v is None:
            out[key] = "null"
        else:
            out[key] = str(v)  # list 등은 문자열로
    return out


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

        # Inverse task self-distillation
        self.inverse_mode = config["training"].get("mode", "forward") == "inverse"
        self.schedule     = config["training"].get("schedule", "none")
        self.w_cycle      = config["training"]["loss_weights"].get("cycle", 0.0)
        self.cycle_start  = config["training"].get("cycle_start", 9999)
        self.w_sketch_dec = config["training"]["loss_weights"].get("sketch_decoder", 0.0)
        self.forward_controlnet: Optional[HairControlNet] = None

        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 2),
            log_with=["tensorboard", "wandb"],
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
        ).to(self.accelerator.device)

        self.ema = EMAModel(
            self.accelerator.unwrap_model(self.controlnet),
            decay=config["training"].get("ema_decay", 0.9999),
        )

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracker 초기화 (TensorBoard + wandb)
        wcfg = config.get("wandb", {})
        run_name = f"{config['training']['phase']}_{config['training']['dataset']}"
        tags = wcfg.get("tags", []) + [config["training"]["phase"], config["training"]["dataset"]]
        self.accelerator.init_trackers(
            project_name=wcfg.get("project", "hair-dit"),
            config=_flatten_config(config),  # TensorBoard은 primitive만 허용
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "entity": wcfg.get("entity") or None,
                    "tags": tags,
                    "dir": str(self.output_dir),
                    "config": config,  # wandb에는 원본 전달
                },
            },
        )

        # Flow matching timestep sampling hyperparams
        self.logit_mean = config["training"].get("logit_mean", 0.0)
        self.logit_std  = config["training"].get("logit_std",  1.0)

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.start_epoch   = 0
        self._current_epoch = 0
        self._restore_training_state()

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
            use_sketch_decoder=cfg["model"].get("use_sketch_decoder", False),
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

        # Inverse mode: load frozen forward ControlNet for cycle self-distillation
        if self.inverse_mode and self.w_cycle > 0.0:
            fwd_ckpt = cfg["training"].get("forward_checkpoint")
            if fwd_ckpt and Path(fwd_ckpt).exists():
                fwd_controlnet = HairControlNet(
                    model_id=model_id,
                    vae=self.vae,
                    num_layers=num_layers,
                    local_files_only=local_files_only,
                )
                ckpt = torch.load(fwd_ckpt, map_location="cpu", weights_only=True)
                fwd_controlnet.load_state_dict(ckpt["controlnet"])
                for p in fwd_controlnet.parameters():
                    p.requires_grad_(False)
                fwd_controlnet.eval()
                self.forward_controlnet = fwd_controlnet
                self.accelerator.print(f"[Cycle] Loaded frozen forward ControlNet from {fwd_ckpt}")
            else:
                self.accelerator.print("[Cycle] forward_checkpoint not found — cycle loss disabled")

        # Phase 2 weight transfer (Phase 1 → Phase 2, controlnet only)
        resume_from = cfg["training"].get("resume_from")
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=True)
            self.controlnet.load_state_dict(ckpt["controlnet"])
            self.accelerator.print(f"Loaded Phase 1 weights from {resume_from}")

        # Full resume: load controlnet weights early (optimizer init needs correct params)
        resume = cfg["training"].get("resume")
        if resume and Path(resume).exists():
            ckpt = torch.load(resume, map_location="cpu", weights_only=True)
            self.controlnet.load_state_dict(ckpt["controlnet"])

    def _setup_data(self):
        cfg = self.cfg["training"]
        aug = build_augmentation_pipeline(self.phase)

        dataset_name = cfg["dataset"]
        if dataset_name == "both":
            from torch.utils.data import ConcatDataset
            train_ds = ConcatDataset([
                HairRegionDataset(split="unbraid_train", augmentation=aug),
                HairRegionDataset(split="braid_train",   augmentation=aug),
            ])
            val_ds = ConcatDataset([
                HairRegionDataset(split="unbraid_test"),
                HairRegionDataset(split="braid_test"),
            ])
        else:
            train_ds = HairRegionDataset(split=f"{dataset_name}_train", augmentation=aug)
            val_ds   = HairRegionDataset(split=f"{dataset_name}_test")

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

        self.accelerator.print(
            f"Starting {self.phase} training for {epochs} epochs"
            + (f" (resuming from epoch {self.start_epoch})" if self.start_epoch else "")
        )

        for epoch in range(self.start_epoch, epochs):
            self._current_epoch = epoch + 1
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
                self.accelerator.log(
                    {"val_loss": val_loss, "epoch": epoch + 1},
                    step=self.global_step,
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pth")

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth")

        self._save_checkpoint("final.pth")
        self.accelerator.end_training()

    def _train_step(self, batch: dict, grad_clip: float = 1.0) -> tuple[torch.Tensor, dict]:
        matte = batch["matte"]   # (B, 1, 512, 512) [0,1]

        # Forward mode: sketch → hair  |  Inverse mode: hair → sketch
        if self.inverse_mode:
            cond_image  = batch["img"]      # full hair photo as conditioning signal
            target      = batch["sketch"]   # generate sketch
            # Weight flow loss on stroke regions (non-black sketch pixels)
            stroke_mask = (target.mean(dim=1, keepdim=True) > 0.05).float()
            loss_mask   = stroke_mask
        else:
            cond_image  = batch["sketch"]   # sketch as conditioning signal
            target      = batch["target"]   # generate hair region (img × matte)
            loss_mask   = matte

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

            # 4. Spatial loss weight at latent resolution
            matte_latent = resize_matte_to_latent(loss_mask)  # (B, 1, 64, 64)

            # Cast inputs to bfloat16 for model forward passes
            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            sigmas_1d = sigmas.view(B).to(dtype=torch.bfloat16)  # (B,) transformer expects bf16

            # 5. ControlNet forward → block_samples + null conditioning
            #    HairControlNet VAE-encodes cond_image as its conditioning signal.
            #    In inverse mode cond_image = hair photo; in forward mode = sketch.
            block_samples, null_enc_hs, null_pooled = self.controlnet(
                noisy_latent=noisy_latents,
                sketch=cond_image,
                matte=matte,
                sigmas=sigmas_1d,
            )
            block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
            if self.schedule != "none":
                block_samples = gate_block_samples(block_samples, matte, self.schedule)
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

            # 8. Direct supervision loss
            total_loss, log_dict = self.loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                matte_latent=matte_latent.to(dtype=torch.bfloat16),
                x_t=noisy_latents,
                sigmas=sigmas.to(dtype=torch.bfloat16),
                vae=self.vae,
                target_rgb=target,
                sketch=cond_image,   # edge loss uses this; disabled in inverse via w_edge=0.0
                matte=loss_mask,
                current_step=self.global_step,
                total_steps=self.total_steps,
            )

            controlnet_unwrapped = self.accelerator.unwrap_model(self.controlnet)

            # 9. Cycle loss: decoder head (O(1)) preferred over full-diffusion fallback
            if self._current_epoch >= self.cycle_start and self.w_cycle > 0.0:
                if (self.inverse_mode
                        and controlnet_unwrapped.sketch_decoder is not None):
                    # Decoder head cycle: block_samples → sketch_pred (no diffusion)
                    sketch_pred_cycle = controlnet_unwrapped.sketch_decoder(block_samples)
                    sketch_gt_cycle   = batch["sketch"].to(device=device, dtype=torch.float32)
                    stroke_cycle = (sketch_gt_cycle.mean(dim=1, keepdim=True) > 0.05).float()
                    cycle_loss = (
                        F.l1_loss(sketch_pred_cycle.float(), sketch_gt_cycle, reduction="none")
                        * stroke_cycle
                    ).sum() / stroke_cycle.sum().clamp(min=1)
                    total_loss = total_loss + self.w_cycle * cycle_loss
                    log_dict["loss_cycle"] = cycle_loss.item()

                elif self.forward_controlnet is not None:
                    # Full-diffusion cycle: x_0 pred → VAE decode → forward ControlNet features
                    x0_pred = noisy_latents.float() - sigmas.float() * v_pred.float()
                    sketch_pred_rgb = self.vae.decode(x0_pred.to(dtype=torch.bfloat16)).clamp(0, 1)
                    block_pred = self.forward_controlnet._get_features_impl(
                        sketch_pred_rgb, matte.to(device=device, dtype=torch.bfloat16)
                    )
                    with torch.no_grad():
                        block_gt = self.forward_controlnet._get_features_impl(
                            batch["sketch"].to(device=device, dtype=torch.bfloat16),
                            matte.to(device=device, dtype=torch.bfloat16),
                        )
                    cycle_loss = sum(
                        F.mse_loss(bp, bg.detach())
                        for bp, bg in zip(block_pred, block_gt)
                    ) / len(block_pred)
                    total_loss = total_loss + self.w_cycle * cycle_loss
                    log_dict["loss_cycle"] = cycle_loss.item()

            # 10. Sketch decoder reconstruction loss (forward mode only)
            #     Forces ControlNet block features to encode sketch structure.
            if (not self.inverse_mode
                    and self.w_sketch_dec > 0.0
                    and controlnet_unwrapped.sketch_decoder is not None):
                sketch_pred_dec = controlnet_unwrapped.sketch_decoder(block_samples)
                sketch_gt_dec   = cond_image.to(dtype=torch.float32)
                stroke_dec = (sketch_gt_dec.mean(dim=1, keepdim=True) > 0.05).float()
                l_sketch_recon = (
                    F.l1_loss(sketch_pred_dec.float(), sketch_gt_dec, reduction="none")
                    * stroke_dec
                ).sum() / stroke_dec.sum().clamp(min=1)
                total_loss = total_loss + self.w_sketch_dec * l_sketch_recon
                log_dict["loss_sketch_recon"] = l_sketch_recon.item()

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
            if self.schedule != "none":
                block_samples = gate_block_samples(block_samples, matte, self.schedule)
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

    def _restore_training_state(self):
        """optimizer / ema / lr_scheduler / step / epoch 전체 복원 (동일 학습 재개용)."""
        resume = self.cfg["training"].get("resume")
        if not resume or not Path(resume).exists():
            return

        ckpt = torch.load(resume, map_location="cpu", weights_only=True)

        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            device = self.accelerator.device
            self.ema.shadow = {k: v.to(device) for k, v in self.ema.shadow.items()}
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # --start_epoch으로 수동 지정 가능 (구 checkpoint에 epoch 키 없을 때)
        self.start_epoch   = (
            self.cfg["training"].get("start_epoch")
            or ckpt.get("epoch", 0)
        )

        self.accelerator.print(
            f"Resumed from {resume} "
            f"(epoch {self.start_epoch}, step {self.global_step})"
        )

    def _save_checkpoint(self, filename: str):
        if not self.accelerator.is_main_process:
            return
        controlnet_unwrapped = self.accelerator.unwrap_model(self.controlnet)
        ckpt = {
            "controlnet":   controlnet_unwrapped.state_dict(),
            "ema":          self.ema.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step":  self.global_step,
            "epoch":        self._current_epoch,
            "best_val_loss": self.best_val_loss,
            "config":       self.cfg,
        }
        save_path = self.output_dir / filename
        torch.save(ckpt, save_path)
        self.accelerator.print(f"Saved checkpoint: {save_path}")
