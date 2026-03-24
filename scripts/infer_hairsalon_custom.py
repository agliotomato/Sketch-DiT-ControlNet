"""
SketchHairSalon(GAN) 커스텀 스케치 추론 스크립트.

직접 그린 sketch 파일 하나만 넣으면 결과 생성.
matte 없으면 sketch의 비검정 영역을 matte로 자동 사용.
color_coding 적용: stroke 색 → 원본 이미지 없으므로 stroke 색 그대로 사용.

Usage:
  python scripts/infer_hairsalon_custom.py \
    --sketch custom/my_sketch.png \
    --style braid \
    --output_dir custom_results/

  # matte 직접 지정
  python scripts/infer_hairsalon_custom.py \
    --sketch custom/my_sketch.png \
    --matte  custom/my_matte.png \
    --style  unbraid \
    --output_dir custom_results/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

HAIRSALON_ROOT = Path(__file__).parent.parent.parent / "SketchHairSalon"
sys.path.insert(0, str(HAIRSALON_ROOT))

from models.Unet_At_Bg import UnetAtBgGenerator

CKPT_ROOT = Path(__file__).parent.parent / "checkpoints"


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


def sketch_to_matte(sketch_rgb: np.ndarray) -> np.ndarray:
    """stroke 내부를 채워서 matte 생성. (H,W) uint8 [0,255]"""
    gray = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # morphological closing으로 stroke 사이 빈 공간 채우기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    matte = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return matte


@torch.no_grad()
def run_s2i(model, sk_matte: np.ndarray, img_rgb: np.ndarray, matte_gray: np.ndarray, device) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    # 모델은 256×256만 처리
    sk_matte_256 = cv2.resize(sk_matte,  (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb_256  = cv2.resize(img_rgb,   (256, 256), interpolation=cv2.INTER_LINEAR)
    matte_256    = cv2.resize(matte_gray,(256, 256), interpolation=cv2.INTER_LINEAR)

    matte_3 = cv2.cvtColor(matte_256, cv2.COLOR_GRAY2RGB)
    noise = generate_noise(256, 256)

    N  = tf.to_tensor(noise).unsqueeze(0).to(device)
    M  = tf.to_tensor(matte_3).unsqueeze(0).to(device)
    SK = (tf.to_tensor(sk_matte_256) * 2.0 - 1.0).unsqueeze(0).to(device)
    IM = (tf.to_tensor(img_rgb_256)  * 2.0 - 1.0).unsqueeze(0).to(device)

    out = model(SK, IM, M, N)
    result_256 = ((out[0] + 1) / 2 * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    result = cv2.resize(result_256, (W, H), interpolation=cv2.INTER_LINEAR)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch",     required=True, help="입력 sketch 이미지 경로")
    parser.add_argument("--img",        default=None,  help="배경 원본 이미지 경로 (없으면 더미 사용)")
    parser.add_argument("--matte",      default=None,  help="matte 경로 (없으면 sketch 비검정 영역 자동 사용)")
    parser.add_argument("--style",      choices=["braid", "unbraid"], default="braid")
    parser.add_argument("--output_dir", default="custom_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_s2i(args.style, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sketch_path = Path(args.sketch)
    stem = sketch_path.stem

    sketch_rgb_orig = np.array(Image.open(sketch_path).convert("RGB"))

    # matte 처리 (matte 또는 img 크기를 기준으로 resize)
    if args.matte:
        matte_gray = np.array(Image.open(args.matte).convert("L"))
        H, W = matte_gray.shape[:2]
    elif args.img:
        ref = np.array(Image.open(args.img).convert("RGB"))
        H, W = ref.shape[:2]
        matte_gray = None
    else:
        H, W = sketch_rgb_orig.shape[:2]
        matte_gray = None

    # sketch를 기준 크기에 맞춤
    sketch_rgb = cv2.resize(sketch_rgb_orig, (W, H), interpolation=cv2.INTER_NEAREST)

    if matte_gray is None:
        print("matte 없음 → sketch 비검정 영역으로 자동 생성")
        matte_gray = sketch_to_matte(sketch_rgb)

    # 배경 이미지
    if args.img:
        img_rgb = np.array(Image.open(args.img).convert("RGB"))
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask = matte_gray > 10
        img_rgb[mask] = 255

    # sk_matte: matte 배경 위에 sketch stroke 올리기
    matte_3 = cv2.cvtColor(matte_gray, cv2.COLOR_GRAY2RGB)
    sk_gray = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2GRAY)
    sk_matte = matte_3.copy()
    sk_matte[sk_gray != 0] = sketch_rgb[sk_gray != 0]

    # 추론
    result = run_s2i(model, sk_matte, img_rgb, matte_gray, device)

    # matte 적용해서 hair patch만 추출
    H_out, W_out = result.shape[:2]
    matte_out = cv2.resize(matte_gray, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
    matte_f = matte_out.astype(np.float32) / 255.0
    hair_patch = (result.astype(np.float32) * matte_f[..., None]).astype(np.uint8)

    # 저장
    Image.fromarray(hair_patch).save(output_dir / f"{stem}_gen.png")
    Image.fromarray(sk_matte).save(output_dir / f"{stem}_sk.png")

    panel = np.concatenate([sketch_rgb, sk_matte, hair_patch], axis=1)
    Image.fromarray(panel).save(output_dir / f"{stem}_panel.png")

    print(f"\n완료: {output_dir}/")
    print(f"  {stem}_gen.png   — 생성 결과")
    print(f"  {stem}_sk.png    — 모델 입력 sketch")
    print(f"  {stem}_panel.png — [원본 sketch | 입력 sketch | 생성 결과]")


if __name__ == "__main__":
    main()
