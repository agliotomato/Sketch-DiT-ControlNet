"""
전체 결과 그리드 시각화
각 샘플(braid_XXXX)당 5열:
  face(원본) | matte | generated hair(DiT) | composite | final
출력: results/grid_overview.png
"""
import os
import glob
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLS = [
    ("face",      "dataset/braid/img/test/{id}.png"),
    ("matte",     "dataset/braid/matte/test/{id}.png"),
    ("generated", "results/DiT(after)/{id}_gen.png"),
    ("composite", "results/composite/{id}.png"),
    ("final",     "results/final/{id}.png"),
]

CELL_SIZE = 256   # 각 셀 크기 (정사각형)
LABEL_H   = 20    # 열 헤더 높이
ROW_LABEL_W = 80  # 행 id 표시 너비
PADDING   = 4


def load_or_placeholder(path, size):
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    # 회색 placeholder
    ph = Image.new("RGB", (size, size), (180, 180, 180))
    draw = ImageDraw.Draw(ph)
    draw.line([(0, 0), (size, size)], fill=(150, 150, 150), width=2)
    draw.line([(size, 0), (0, size)], fill=(150, 150, 150), width=2)
    return ph


def collect_ids():
    pattern = os.path.join(BASE, "results", "final", "braid_*.png")
    paths = glob.glob(pattern)
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)
    return ids


def make_header(n_cols, col_labels):
    w = ROW_LABEL_W + n_cols * (CELL_SIZE + PADDING)
    h = LABEL_H
    img = Image.new("RGB", (w, h), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(col_labels):
        x = ROW_LABEL_W + i * (CELL_SIZE + PADDING) + CELL_SIZE // 2
        draw.text((x, 2), label, fill=(255, 255, 255), anchor="mt")
    return img


def make_row(sample_id):
    cells = []
    for _, path_tmpl in COLS:
        path = os.path.join(BASE, path_tmpl.format(id=sample_id))
        cells.append(load_or_placeholder(path, CELL_SIZE))

    row_w = ROW_LABEL_W + len(cells) * (CELL_SIZE + PADDING)
    row_h = CELL_SIZE
    row = Image.new("RGB", (row_w, row_h), (30, 30, 30))

    # 행 id 라벨
    draw = ImageDraw.Draw(row)
    label = sample_id.replace("braid_", "")
    draw.text((ROW_LABEL_W // 2, CELL_SIZE // 2), label,
              fill=(220, 220, 220), anchor="mm")

    for i, cell in enumerate(cells):
        x = ROW_LABEL_W + i * (CELL_SIZE + PADDING)
        row.paste(cell, (x, 0))

    return row


def main():
    ids = collect_ids()
    if not ids:
        print("results/final/ 에 braid_*.png 파일이 없습니다.")
        return

    print(f"샘플 {len(ids)}개 발견: {ids}")

    col_labels = [c[0] for c in COLS]
    header = make_header(len(COLS), col_labels)
    rows = [header]
    for sid in ids:
        rows.append(make_row(sid))
        print(f"  {sid} 처리 완료")

    total_w = rows[0].width
    total_h = sum(r.height for r in rows) + PADDING * (len(rows) - 1)

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + PADDING

    out_path = os.path.join(BASE, "results", "grid_overview.png")
    canvas.save(out_path)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
