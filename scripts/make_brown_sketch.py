from PIL import Image
import numpy as np

BROWN = (180, 120, 50)
THRESHOLD = 80  # 높은 threshold로 어두운 blob 제거, 밝은 스트록만 유지

def process_sketch(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    brightness = arr.sum(axis=2)
    mask = brightness > THRESHOLD
    result = np.zeros_like(arr)
    result[mask] = BROWN
    # 워터마크(우측 하단 다이아몬드) 제거: 마지막 50x50 픽셀 강제 블랙
    result[-50:, -50:] = (0, 0, 0)
    out_img = Image.fromarray(result.astype(np.uint8))
    out_img.save(output_path)
    print(f'Saved: {output_path}  |  stroke pixels: {mask.sum()}')

base = '/home/agliotomato/hair-dit/dataset/braid/sketch/test'
files = [
    (f'{base}/limit_sketch.png',  f'{base}/limit_sketch_brown.png'),
    (f'{base}/limit2_sketch.png', f'{base}/limit2_sketch_brown.png'),
]
for inp, out in files:
    process_sketch(inp, out)
