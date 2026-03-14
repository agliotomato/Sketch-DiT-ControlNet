"""
phase2.md의 Results 섹션에 16샘플 이미지 표를 추가합니다.
target = img * matte (soft composite) 이미지를 results/ 에 저장합니다.

Usage:
  python scripts/make_report.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import HairRegionDataset

RESULTS_DIR = Path("results")
REPORT_PATH = Path("phase2.md")
NUM_SAMPLES = 16


def main():
    dataset = HairRegionDataset(split="braid_test")
    n = min(NUM_SAMPLES, len(dataset))

    RESULTS_DIR.mkdir(exist_ok=True)

    rows = ["| Sketch | Matte | Generated | Target |",
            "|--------|-------|-----------|--------|"]

    for idx in range(n):
        stem = dataset.stems[idx]
        data = dataset[idx]

        # target = img * matte (soft composite) 저장
        target_t = data["target"]  # (3, H, W) float [0,1]
        target_np = (target_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target_save = RESULTS_DIR / f"{idx:04d}_target.png"
        Image.fromarray(target_np).save(target_save)

        sketch_path = f"dataset3/braid/sketch/test/{stem}.png"
        matte_path  = f"dataset3/braid/matte/test/{stem}.png"
        gen_path    = f"results/{idx:04d}_gen.png"
        target_path = f"results/{idx:04d}_target.png"

        rows.append(
            f"| ![sketch]({sketch_path}) | ![matte]({matte_path}) "
            f"| ![generated]({gen_path}) | ![target]({target_path}) |"
        )

    table = "\n".join(rows)

    section = f"""
---

## Inference 결과 이미지 (braid_test, {n}샘플)

| 열 | 내용 |
|---|---|
| Sketch | 입력 컬러 스케치 (strand 구분용 색) |
| Matte | 입력 soft alpha matte |
| Generated | 모델 출력 (flow matching 20스텝) |
| Target | 정답 (img × matte soft composite) |

{table}
"""

    original = REPORT_PATH.read_text(encoding="utf-8")
    marker = "\n---\n\n## Inference 결과 이미지"
    if marker in original:
        original = original[:original.index(marker)]

    REPORT_PATH.write_text(original + section, encoding="utf-8")
    print(f"target {n}개 저장 완료: {RESULTS_DIR}/")
    print(f"Updated: {REPORT_PATH}")


if __name__ == "__main__":
    main()
