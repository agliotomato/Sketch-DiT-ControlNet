"""
Inverse Head 학습 진입점.

Usage:
  python scripts/train_inverse_head.py --config configs/inverse_head.yaml

  # Multi-GPU:
  accelerate launch scripts/train_inverse_head.py --config configs/inverse_head.yaml

  # Resume:
  python scripts/train_inverse_head.py --config configs/inverse_head.yaml \
      --resume checkpoints/inverse_head/epoch_10.pth
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer_inverse_head import InverseHeadTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--start_epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["training"]["resume"] = args.resume
    if args.start_epoch is not None:
        cfg["training"]["start_epoch"] = args.start_epoch

    trainer = InverseHeadTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
