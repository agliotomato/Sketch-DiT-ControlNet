#!/usr/bin/env python3
"""
스케치/매트 이미지 정제 도구
- sketch: 배경 검정, 스트록 단일 갈색, 워터마크 제거
- matte : 배경 검정(0), 헤어 흰색(255), 워터마크 제거
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


BROWN = (180, 120, 50)
WATERMARK_SIZE = 50  # 우측 하단 제거할 크기(px)


def clean_sketch(arr: np.ndarray, threshold: int) -> np.ndarray:
    mask = arr.sum(axis=2) > threshold
    result = np.zeros_like(arr)
    result[mask] = BROWN
    return result


def clean_matte(arr: np.ndarray, threshold: int) -> np.ndarray:
    gray = arr.mean(axis=2)
    result = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return np.stack([result] * 3, axis=2)


def remove_watermark(arr: np.ndarray, size: int) -> np.ndarray:
    arr[-size:, -size:] = 0
    return arr


def process(input_path: Path, output_path: Path, mode: str, threshold: int, no_watermark: bool):
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)

    if mode == 'sketch':
        result = clean_sketch(arr, threshold)
    elif mode == 'matte':
        result = clean_matte(arr, threshold)
    else:
        raise ValueError(f"mode는 'sketch' 또는 'matte' 중 하나여야 합니다: {mode}")

    if not no_watermark:
        result = remove_watermark(result, WATERMARK_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.astype(np.uint8)).save(output_path)
    print(f"[{mode}] {input_path.name} → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="스케치/매트 이미지 정제")
    parser.add_argument("inputs", nargs="+", type=Path, help="입력 이미지 경로 (여러 개 가능)")
    parser.add_argument("-m", "--mode", choices=["sketch", "matte"], required=True,
                        help="처리 모드: sketch (갈색 단일색) | matte (흑백 이진화)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="출력 디렉토리 (미지정 시 원본과 같은 위치에 _clean 접미사로 저장)")
    parser.add_argument("-t", "--threshold", type=int, default=None,
                        help="픽셀 판별 threshold (sketch 기본값: 80, matte 기본값: 128)")
    parser.add_argument("--no-watermark", action="store_true",
                        help="워터마크 제거 건너뜀")
    parser.add_argument("--overwrite", action="store_true",
                        help="원본 파일 덮어쓰기")
    args = parser.parse_args()

    default_threshold = 80 if args.mode == "sketch" else 128
    threshold = args.threshold if args.threshold is not None else default_threshold

    for inp in args.inputs:
        if not inp.exists():
            print(f"[경고] 파일 없음: {inp}")
            continue

        if args.overwrite:
            out = inp
        elif args.output_dir:
            out = args.output_dir / inp.name
        else:
            out = inp.with_stem(inp.stem + "_clean")

        process(inp, out, args.mode, threshold, args.no_watermark)


if __name__ == "__main__":
    main()
