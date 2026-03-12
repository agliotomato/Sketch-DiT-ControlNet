"""
Single-sample inference: (sketch + matte) → hair region image.

Usage:
  python scripts/infer.py \\
    --checkpoint checkpoints/phase2_braid/best.pth \\
    --sketch path/to/sketch.png \\
    --matte  path/to/matte.png \\
    --output output/hair_region.png \\
    --steps 50
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dit import HairDiT
from src.models.conditioning import ConditionEncoder
from src.models.vae_wrapper import VAEWrapper
from src.training.scheduler import DDPMScheduler


def load_image(path: str, mode: str = "RGB", size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def infer(
    checkpoint_path: str,
    sketch_path: str,
    matte_path: str,
    output_path: str,
    num_steps: int = 50,
    device: str = "cuda",
    image_size: int = 512,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # Init models
    vae = VAEWrapper.from_pretrained().to(device)
    dit = HairDiT(
        input_size=model_cfg.get("input_size", 64),
        patch_size=model_cfg.get("patch_size", 2),
        in_channels=model_cfg.get("in_channels", 4),
        hidden_size=model_cfg.get("hidden_size", 512),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 8),
    ).to(device)
    cond_enc = ConditionEncoder(
        vae=vae,
        hidden_size=model_cfg.get("hidden_size", 512),
    ).to(device)

    dit.load_state_dict(ckpt["dit"])
    cond_enc.matte_cnn.load_state_dict(ckpt["matte_cnn"])
    cond_enc.sketch_patch_embed.load_state_dict(ckpt["sketch_embed"])
    dit.eval()
    cond_enc.eval()

    # Load inputs
    sketch = load_image(sketch_path, "RGB",  image_size).to(device)
    matte  = load_image(matte_path,  "L",    image_size).to(device)

    # Encode conditions
    sketch_tokens, matte_feat = cond_enc(sketch, matte)

    # DDPM sampling
    scheduler = DDPMScheduler(num_timesteps=1000).to(device)
    latent_size = image_size // 8  # 64

    x = torch.randn(1, 4, latent_size, latent_size, device=device)

    # Use subset of timesteps for faster inference
    T = scheduler.num_timesteps
    step_indices = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)

    for t_val in step_indices:
        t_batch = t_val.expand(1)
        noise_pred = dit(x, matte_feat, t_batch, sketch_tokens)
        x = scheduler.sample_step(x, noise_pred, t_batch, eta=1.0)

    # Decode
    hair_region_11 = vae.decode(x)            # (1, 3, 512, 512) [-1, 1]
    hair_region_01 = VAEWrapper.denormalize(hair_region_11).clamp(0, 1)

    # Apply matte mask
    hair_region_01 = hair_region_01 * matte

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(hair_region_01, output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sketch",     required=True)
    parser.add_argument("--matte",      required=True)
    parser.add_argument("--output",     default="output/hair_region.png")
    parser.add_argument("--steps",      type=int, default=50)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    infer(
        checkpoint_path=args.checkpoint,
        sketch_path=args.sketch,
        matte_path=args.matte,
        output_path=args.output,
        num_steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
