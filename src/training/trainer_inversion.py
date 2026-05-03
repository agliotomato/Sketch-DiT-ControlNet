"""
InversionTrainer: Hair→Sketch Inversion Adapter 학습.

- VAE, HairControlNet: frozen (phase2_braid checkpoint 로드)
- HairInversionAdapter: 학습 대상
- Phase A (epoch 0~phase_b_start): structure + color + matte loss
- Phase B (epoch phase_b_start~): + feature cycle consistency loss
"""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import SD3Transformer2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairRegionDataset
from src.models.controlnet_sd35 import HairControlNet
from src.models.inversion_adapter import HairInversionAdapter
from src.models.vae_wrapper import VAEWrapper
from src.training.losses_inversion import InversionLoss


class InversionTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        tcfg = config["training"]

        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            log_with=["tensorboard", "wandb"],
            project_dir=config["checkpointing"]["output_dir"],
        )

        model_id        = config["model"]["model_id"]
        local_only      = config.get("local_files_only", False)
        controlnet_ckpt = config["model"]["controlnet_checkpoint"]

        # --- Frozen models ---
        self.vae = VAEWrapper.from_pretrained(
            model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_only,
        ).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.controlnet = HairControlNet(
            model_id=model_id, vae=self.vae,
            num_layers=config["model"].get("num_controlnet_layers", 12),
            local_files_only=local_only,
        )
        ckpt = torch.load(controlnet_ckpt, map_location="cpu", weights_only=True)
        self.controlnet.load_state_dict(ckpt["controlnet"])
        self.controlnet.eval()
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        # --- Trainable adapter ---
        self.adapter = HairInversionAdapter(
            vae=self.vae,
            grid_size=config.get("grid_size", 16),
        )

        # --- Data ---
        aug = build_augmentation_pipeline("pretrain")  # color sampler 포함
        datasets = []
        for split in ("unbraid_train", "braid_train"):
            datasets.append(HairRegionDataset(split=split, augmentation=aug))
        train_ds = ConcatDataset(datasets)

        val_datasets = []
        for split in ("unbraid_test", "braid_test"):
            val_datasets.append(HairRegionDataset(split=split))
        val_ds = ConcatDataset(val_datasets)

        bs = tcfg.get("batch_size", 8)
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        # --- Optimizer & scheduler ---
        self.optimizer = AdamW(
            self.adapter.parameters(),
            lr=tcfg.get("learning_rate", 1e-4),
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        total_steps = tcfg.get("epochs", 50) * len(self.train_loader)
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
            w_feature=lw.get("feature", 0.1),
        )

        self.phase_b_start          = tcfg.get("phase_b_start", 10)
        self.feature_warmup_epochs  = tcfg.get("feature_warmup_epochs", 10)
        self.w_feature_max          = lw.get("feature", 0.1)
        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Accelerate prepare ---
        (
            self.adapter,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.adapter, self.optimizer,
            self.train_loader, self.val_loader,
            self.lr_scheduler,
        )
        device = self.accelerator.device
        self.vae        = self.vae.to(device)
        self.controlnet = self.controlnet.to(device)

        wcfg = config.get("wandb", {})
        self.accelerator.init_trackers(
            project_name=wcfg.get("project", "hair-dit"),
            config=_flatten(config),
            init_kwargs={"wandb": {
                "name": "inversion_adapter",
                "entity": wcfg.get("entity") or None,
                "tags": ["inversion"],
                "config": config,
            }},
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------

    def train(self):
        tcfg   = self.cfg["training"]
        epochs = tcfg.get("epochs", 50)
        save_every = self.cfg["checkpointing"].get("save_every", 10)
        eval_every = self.cfg["checkpointing"].get("eval_every", 5)

        for epoch in range(epochs):
            phase_b = (epoch >= self.phase_b_start)

            # feature loss warm-up: phase_b 진입 후 feature_warmup_epochs에 걸쳐 0→w_feature_max
            if phase_b:
                warmup_frac = min(1.0, (epoch - self.phase_b_start) / max(self.feature_warmup_epochs, 1))
                w_feature_current = self.w_feature_max * warmup_frac
            else:
                w_feature_current = 0.0

            self.adapter.train()
            epoch_losses = []

            desc = f"Epoch {epoch+1}/{epochs}"
            if phase_b:
                desc += f" [feat w={w_feature_current:.3f}]"
            progress = tqdm(
                self.train_loader, desc=desc,
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress:
                loss, log_dict = self._train_step(batch, phase_b=phase_b, w_feature_current=w_feature_current)
                epoch_losses.append(log_dict["loss_total"])

                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.cfg["training"]["learning_rate"] * lr_scale
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
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best.pth", epoch)

            if (epoch + 1) % save_every == 0:
                self._save(f"epoch_{epoch+1}.pth", epoch)

        self._save("final.pth", epochs - 1)
        self.accelerator.end_training()

    def _train_step(self, batch: dict, phase_b: bool, w_feature_current: float = 0.0) -> tuple[torch.Tensor, dict]:
        device = self.accelerator.device
        dtype  = torch.bfloat16

        hair_image  = batch["target"].to(device, dtype=dtype)   # (B, 3, 512, 512) img*matte
        sketch_gt   = batch["sketch"].to(device, dtype=dtype)   # (B, 3, 512, 512)
        matte_gt    = batch["matte"].to(device, dtype=dtype)    # (B, 1, 512, 512)

        self.optimizer.zero_grad()
        sketch_pred, matte_pred, stroke_mask = self.adapter(hair_image)

        block_pred = block_gt = None
        if phase_b:
            block_pred = self.controlnet.get_features(sketch_pred.float(), matte_pred.float())
            block_gt   = self.controlnet.get_features(sketch_gt.float(),   matte_gt.float())

        loss, log_dict = self.loss_fn(
            stroke_mask_pred=stroke_mask,
            sketch_pred=sketch_pred,
            matte_pred=matte_pred,
            sketch_gt=sketch_gt,
            matte_gt=matte_gt,
            block_pred=block_pred,
            block_gt=block_gt,
            w_feature_current=w_feature_current,
        )

        self.accelerator.backward(loss)
        if self.cfg["training"].get("gradient_clip"):
            self.accelerator.clip_grad_norm_(self.adapter.parameters(), self.cfg["training"]["gradient_clip"])
        self.optimizer.step()
        return loss, log_dict

    @torch.no_grad()
    def _validate(self) -> float:
        self.adapter.eval()
        device = self.accelerator.device
        dtype  = torch.bfloat16
        total, n = 0.0, 0

        for batch in self.val_loader:
            hair_image = batch["target"].to(device, dtype=dtype)
            sketch_gt  = batch["sketch"].to(device, dtype=dtype)
            matte_gt   = batch["matte"].to(device, dtype=dtype)

            sketch_pred, matte_pred, stroke_mask = self.adapter(hair_image)
            _, log = self.loss_fn(stroke_mask, sketch_pred, matte_pred, sketch_gt, matte_gt)
            total += log["loss_total"]
            n += 1

        return total / max(n, 1)

    def _save(self, filename: str, epoch: int):
        if not self.accelerator.is_main_process:
            return
        ckpt = {
            "adapter":      self.accelerator.unwrap_model(self.adapter).state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "global_step":  self.global_step,
            "epoch":        epoch + 1,
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
