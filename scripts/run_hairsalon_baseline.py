"""
SketchHairSalon(GAN) baseline 결과를 우리 test 데이터셋에서 생성.

우리 ground truth matte를 그대로 사용해서 공정한 비교를 위해
S2M(Sketch→Matte) 단계는 건너뜀.

출력: checkpoints/hairsalon_results/{braid|unbraid}_test/{stem}.png
      (full image에서 matte로 crop한 hair patch)

Usage:
  cd /home/agliotomato/hair-dit
  python scripts/run_hairsalon_baseline.py --style braid
  python scripts/run_hairsalon_baseline.py --style unbraid

  # color_coding 적용 (논문 원래 방식: stroke 색 → target 실제 픽셀 색으로 교체)
  python scripts/run_hairsalon_baseline.py --style braid --color_code
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

HAIRSALON_ROOT = Path(__file__).parent.parent.parent / "SketchHairSalon"
sys.path.insert(0, str(HAIRSALON_ROOT))

from models.Unet_At_Bg import UnetAtBgGenerator
from data.color_coding import color_coding

DATASET_ROOT = Path(__file__).parent.parent / "dataset3"
CKPT_ROOT    = Path(__file__).parent.parent / "checkpoints"


def load_s2i(style: str, device: torch.device) -> UnetAtBgGenerator:
    if style == "braid":
        ckpt_path = CKPT_ROOT / "hairsalon_S2I_braid" / "400_net_G.pth"
    else:
        ckpt_path = CKPT_ROOT / "hairsalon_S2I_unbraid" / "200_net_G.pth"

    model = UnetAtBgGenerator(3, 3, 8, 64, use_dropout=False)
    state_dict = torch.load(ckpt_path, map_location=str(device))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded S2I ({style}): {ckpt_path}")
    return model


def generate_noise(width: int, height: int) -> np.ndarray:
    weight, weightSum = 1.0, 0.0
    noise = np.zeros((height, width, 3), dtype=np.float32)
    w, h = width, height
    while w >= 8 and h >= 8:
        noise += cv2.resize(
            np.random.normal(0.5, 0.25, (int(h), int(w), 3)),
            dsize=(noise.shape[1], noise.shape[0]),
        ) * weight
        weightSum += weight
        w //= 2
        h //= 2
    return noise / weightSum


@torch.no_grad()
def run_s2i(model, sk_matte: np.ndarray, img_rgb: np.ndarray, matte_gray: np.ndarray, device) -> np.ndarray:
    """
    sk_matte:    (H,W,3) uint8  matte background + colored strokes (모델 입력 형태)
    img_rgb:     (H,W,3) uint8  original face image
    matte_gray:  (H,W)   uint8  soft matte [0,255]
    Returns:     (H,W,3) uint8  generated full image
    """
    H, W = img_rgb.shape[:2]
    matte_3 = cv2.cvtColor(matte_gray, cv2.COLOR_GRAY2RGB)  # (H,W,3)

    noise = generate_noise(W, H)

    N  = tf.to_tensor(noise).unsqueeze(0).to(device)
    M  = tf.to_tensor(matte_3).unsqueeze(0).to(device)
    SK = (tf.to_tensor(sk_matte) * 2.0 - 1.0).unsqueeze(0).to(device)
    IM = (tf.to_tensor(img_rgb)  * 2.0 - 1.0).unsqueeze(0).to(device)

    out = model(SK, IM, M, N)  # (1,3,H,W) in [-1,1]
    result = ((out[0] + 1) / 2 * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return result


def make_sk_matte(sketch_rgb: np.ndarray, img_rgb: np.ndarray, matte_gray: np.ndarray,
                  use_color_coding: bool) -> np.ndarray:
    """colored sketch를 matte background 위에 올린 sk_matte 생성."""
    matte_3 = cv2.cvtColor(matte_gray, cv2.COLOR_GRAY2RGB)
    if use_color_coding:
        # 논문 원래 방식: grayscale stroke 값으로 구분 → target 픽셀 색으로 교체
        sketch_gray = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2GRAY)
        sk_matte = color_coding(img_rgb, sketch_gray, matte_3)
    else:
        # 기존 방식: 임의 레이블 색 그대로 사용
        sk_matte = matte_3.copy()
        sk_gray = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2GRAY)
        sk_matte[sk_gray != 0] = sketch_rgb[sk_gray != 0]
    return sk_matte


def process(style: str, subset: str, model, device, use_color_coding: bool,
            max_samples: int = 0, save_input: bool = False):
    base = DATASET_ROOT / style
    img_dir    = base / "img"    / subset
    sketch_dir = base / "sketch" / subset
    matte_dir  = base / "matte"  / subset
    suffix  = "_color_coded" if use_color_coding else ""
    out_dir = CKPT_ROOT / "hairsalon_results" / f"{style}_{subset}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if max_samples > 0:
        stems = stems[:max_samples]
    print(f"\n{style}/{subset}: {len(stems)} samples → {out_dir}")

    for stem in tqdm(stems, desc=f"{style}/{subset}"):
        out_path = out_dir / f"{stem}.png"
        if out_path.exists() and not save_input:
            continue

        img_rgb    = np.array(Image.open(img_dir    / f"{stem}.png").convert("RGB"))
        sketch_rgb = np.array(Image.open(sketch_dir / f"{stem}.png").convert("RGB"))
        matte_gray = np.array(Image.open(matte_dir  / f"{stem}.png").convert("L"))

        sk_matte = make_sk_matte(sketch_rgb, img_rgb, matte_gray, use_color_coding)

        if save_input:
            Image.fromarray(sk_matte).save(out_dir / f"{stem}_sk.png")

        if not out_path.exists():
            full_img = run_s2i(model, sk_matte, img_rgb, matte_gray, device)

            # hair patch = full_img × matte
            matte_f = matte_gray.astype(np.float32) / 255.0
            hair_patch = (full_img.astype(np.float32) * matte_f[..., None]).astype(np.uint8)

            Image.fromarray(hair_patch).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style",      choices=["braid", "unbraid"], default="braid")
    parser.add_argument("--subset",     choices=["train", "test", "both"], default="test")
    parser.add_argument("--color_code",  action="store_true",
                        help="논문 원래 방식: stroke 색을 target 실제 픽셀 색으로 교체 후 입력")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="처리할 최대 샘플 수 (0=전체)")
    parser.add_argument("--save_input",  action="store_true",
                        help="모델 입력(sk_matte)을 원본 스케치와 비교 패널로 저장")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_s2i(args.style, device)

    subsets = ["train", "test"] if args.subset == "both" else [args.subset]
    for subset in subsets:
        process(args.style, subset, model, device, args.color_code, args.max_samples, args.save_input)

    print(f"\n완료: checkpoints/hairsalon_results/")


if __name__ == "__main__":
    main()
