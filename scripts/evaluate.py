"""
Full evaluation pipeline.

Runs all metrics (SHR, MCS, LPIPS, FID, BSS) on a given split.

Usage:
  python scripts/evaluate.py \\
    --checkpoint checkpoints/phase2_braid/best.pth \\
    --split braid_test \\
    --output_dir eval_results/braid/
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import HairRegionDataset
from src.models.dit import HairDiT
from src.models.conditioning import ConditionEncoder
from src.models.vae_wrapper import VAEWrapper
from src.training.scheduler import DDPMScheduler
from src.evaluation.sketch_eval import SketchEvaluator
from src.evaluation.visualize import save_result_grid


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    split: str,
    output_dir: str,
    num_inference_steps: int = 50,
    device: str = "cuda",
    batch_size: int = 4,
    num_vis_samples: int = 20,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint & models
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    vae = VAEWrapper.from_pretrained().to(device)
    dit = HairDiT(
        input_size=model_cfg.get("input_size", 64),
        hidden_size=model_cfg.get("hidden_size", 512),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 8),
    ).to(device)
    cond_enc = ConditionEncoder(vae=vae, hidden_size=model_cfg.get("hidden_size", 512)).to(device)

    dit.load_state_dict(ckpt["dit"])
    cond_enc.matte_cnn.load_state_dict(ckpt["matte_cnn"])
    cond_enc.sketch_patch_embed.load_state_dict(ckpt["sketch_embed"])
    dit.eval()
    cond_enc.eval()

    scheduler = DDPMScheduler().to(device)
    evaluator = SketchEvaluator(device=device, is_braid="braid" in split)

    dataset = HairRegionDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_results = []

    for batch in loader:
        sketch = batch["sketch"].to(device)
        matte  = batch["matte"].to(device)
        target = batch["target"].to(device)
        filenames = batch["filename"]

        # Generate hair region
        sketch_tokens, matte_feat = cond_enc(sketch, matte)
        latent_size = 64
        x = torch.randn(len(sketch), 4, latent_size, latent_size, device=device)
        T = scheduler.num_timesteps
        step_indices = torch.linspace(T - 1, 0, num_inference_steps, dtype=torch.long, device=device)

        for t_val in step_indices:
            t_batch = t_val.expand(len(sketch))
            noise_pred = dit(x, matte_feat, t_batch, sketch_tokens)
            x = scheduler.sample_step(x, noise_pred, t_batch, eta=1.0)

        pred_11 = vae.decode(x)
        pred_01 = VAEWrapper.denormalize(pred_11).clamp(0, 1) * matte

        # Compute per-sample metrics
        metrics = evaluator.compute_batch(pred_01, target, sketch, matte, filenames)
        all_results.extend(metrics)

    # Aggregate and save metrics
    evaluator.save_report(all_results, output_dir / "metrics.json")
    evaluator.print_summary(all_results)

    # Visualize worst/best SHR samples
    all_results_sorted = sorted(all_results, key=lambda x: x["shr"])
    worst_names = [r["filename"] for r in all_results_sorted[:num_vis_samples]]
    best_names  = [r["filename"] for r in all_results_sorted[-num_vis_samples:]]
    save_result_grid(worst_names, split, output_dir / "worst_shr_grid.png")
    save_result_grid(best_names,  split, output_dir / "best_shr_grid.png")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--split",         default="braid_test")
    parser.add_argument("--output_dir",    default="eval_results/")
    parser.add_argument("--steps",         type=int, default=50)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--batch_size",    type=int, default=4)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        num_inference_steps=args.steps,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
