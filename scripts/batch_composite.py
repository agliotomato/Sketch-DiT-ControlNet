"""
Batch composite: DiT(after) generated hair regions onto original face images.

  composite = hair_region * matte + face_image * (1 - matte)

Usage:
  python scripts/batch_composite.py
  python scripts/batch_composite.py --max_id 2676
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.composite import composite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",   default="results/DiT(after)")
    parser.add_argument("--img_dir",   default="dataset/braid/img/test")
    parser.add_argument("--matte_dir", default="dataset/braid/matte/test")
    parser.add_argument("--output_dir", default="results/composite")
    parser.add_argument("--max_id",    type=int,   default=2676)
    parser.add_argument("--feather",   type=float, default=3.0, help="Gaussian feathering sigma (0=off)")
    args = parser.parse_args()

    gen_dir   = Path(args.gen_dir)
    img_dir   = Path(args.img_dir)
    matte_dir = Path(args.matte_dir)
    output_dir = Path(args.output_dir)

    gen_files = sorted(gen_dir.glob("*_gen.png"))
    gen_files = [f for f in gen_files if int(f.stem.split("_")[1]) <= args.max_id]

    print(f"Found {len(gen_files)} files (up to braid_{args.max_id})")

    for gen_path in gen_files:
        stem = gen_path.stem.replace("_gen", "")  # e.g. braid_2534
        face_path  = img_dir   / f"{stem}.png"
        matte_path = matte_dir / f"{stem}.png"
        output_path = output_dir / f"{stem}.png"

        if not face_path.exists():
            print(f"  [SKIP] face not found: {face_path}")
            continue
        if not matte_path.exists():
            print(f"  [SKIP] matte not found: {matte_path}")
            continue

        composite(str(gen_path), str(face_path), str(matte_path), str(output_path), feather_sigma=args.feather)


if __name__ == "__main__":
    main()
