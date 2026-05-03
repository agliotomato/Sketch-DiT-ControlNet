"""
Inversion Adapter 학습 entry point.

Usage:
  python scripts/train_inversion.py --config configs/inversion.yaml
  accelerate launch scripts/train_inversion.py --config configs/inversion.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer_inversion import InversionTrainer


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = deep_merge(base, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Inversion adapter training | controlnet: {cfg['model']['controlnet_checkpoint']}")

    trainer = InversionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
