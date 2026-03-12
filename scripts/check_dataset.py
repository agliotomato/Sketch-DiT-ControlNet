"""
Dataset integrity checker.
Verifies all (img, sketch, matte) triplets exist and are loadable.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image


DATASET_ROOT = Path(__file__).parent.parent / "dataset3"

SPLITS = {
    "braid_train":   ("braid",   "train"),
    "braid_test":    ("braid",   "test"),
    "unbraid_train": ("unbraid", "train"),
    "unbraid_test":  ("unbraid", "test"),
}


def check_split(style: str, subset: str) -> tuple[int, list[str]]:
    base = DATASET_ROOT / style
    img_dir    = base / "img"    / subset
    sketch_dir = base / "sketch" / subset
    matte_dir  = base / "matte"  / subset

    for d in [img_dir, sketch_dir, matte_dir]:
        if not d.exists():
            return 0, [f"MISSING directory: {d}"]

    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    errors = []

    for stem in stems:
        img_path    = img_dir    / f"{stem}.png"
        sketch_path = sketch_dir / f"{stem}.png"
        matte_path  = matte_dir  / f"{stem}.png"

        for path, label in [(sketch_path, "sketch"), (matte_path, "matte")]:
            if not path.exists():
                errors.append(f"MISSING {label}: {path}")
                continue
            try:
                img = Image.open(path)
                img.verify()
            except Exception as e:
                errors.append(f"CORRUPT {label} {path}: {e}")

        # check img too
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            errors.append(f"CORRUPT img {img_path}: {e}")

    return len(stems), errors


def main():
    total_ok = 0
    all_errors = []

    for split_name, (style, subset) in SPLITS.items():
        n, errors = check_split(style, subset)
        status = "OK" if not errors else f"{len(errors)} ERRORS"
        print(f"[{split_name:15s}] {n:5d} samples — {status}")
        total_ok += n - len(errors)
        all_errors.extend(errors)

    print()
    if all_errors:
        print(f"=== {len(all_errors)} errors found ===")
        for e in all_errors[:50]:
            print(" ", e)
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more")
        sys.exit(1)
    else:
        print(f"All triplets valid. Total samples: {total_ok}")


if __name__ == "__main__":
    main()
