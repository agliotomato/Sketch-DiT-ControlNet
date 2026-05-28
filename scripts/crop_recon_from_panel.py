"""
패널 이미지에서 4번째 열(hair_recon)만 잘라 저장.
Panel 구조: [hair_orig | sketch_pred | matte | hair_recon] — 각 512×512
"""

import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel_dir",  default="outputs/eval_roundtrip/")
    parser.add_argument("--output_dir", default="roundtrip_onlyhair/")
    parser.add_argument("--col",        type=int, default=3, help="0-indexed column (3=hair_recon)")
    parser.add_argument("--size",       type=int, default=512)
    args = parser.parse_args()

    panel_dir  = Path(args.panel_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = sorted(panel_dir.glob("*_panel.png"))
    if not panels:
        print(f"No *_panel.png found in {panel_dir}")
        return

    col_w = args.size
    x0 = args.col * col_w
    x1 = x0 + col_w

    for p in panels:
        img  = Image.open(p)
        crop = img.crop((x0, 0, x1, img.height))
        if crop.size != (args.size, args.size):
            crop = crop.resize((args.size, args.size), Image.LANCZOS)
        out_path = output_dir / p.name.replace("_panel", "")
        crop.save(out_path)
        print(f"  {p.name} → {out_path.name}")

    print(f"\n{len(panels)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
