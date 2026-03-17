# Stroke 색상의 역할 및 학습/추론 방식 정리

---

## 1. 스케치 stroke 색상의 의미

학습 데이터의 스케치에서 볼 수 있는 알록달록한 선(보라색, 파란색, 초록색 등)은 **시각화 및 인스턴스 구분용**이다.

- 논문 Figure 등에서 개별 머리 가닥을 구분해서 보여주기 위한 임시 색상
- 실제 딥러닝 모델의 최종 입력값이 아님(중요)
- 각 stroke마다 서로 다른 임의 레이블 색이 부여되어 있음

**① 구조 구분용 레이블**

인접한 머리 가닥들을 개별적으로 식별하기 위해 서로 다른 색을 부여한다.
머리카락은 겹치고 꼬이는 구조가 복잡하기 때문에, 시각적으로 명확히 구분하기 위한 임시 표현이다.
이 색들은 실제 머리 색과 무관한 임의 레이블이다.(랜덤)

**② 사용자 색상 제어 (hair dyeing)** -- 추론시 사용 

시스템 인터페이스에서 사용자가 원하는 색으로 선을 그리면 그 색에 맞는 머리가 생성된다.
SketchHairSalon은 7만개 이상의 실제 머리 색 DB를 활용해, 사용자가 선택한 색과 가장 가까운 자연스러운 머리 색으로 매핑하여 사용한다.

---

## 2. SketchHairSalon (GAN) 방식

### stroke 구분 방법

GAN의 `color_coding` 함수는 grayscale 입력을 요구하도록 구현되어 있어, RGB 스케치를 먼저 grayscale로 변환한 뒤 밝기 값으로 stroke를 구분한다.

```python
sketch_gray = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2GRAY)
sk_matte = color_coding(img_rgb, sketch_gray, matte_3)
```

단점: 서로 다른 색의 stroke라도 밝기가 비슷하면 같은 stroke로 합쳐지는 collision 위험이 있다.

### 학습 시

스케치의 각 stroke는 임의 레이블 색이지만, **S2I-Net 학습 시 실제 머리 픽셀 색으로 교체**된다.

```python
# augment=True 시 (학습)
if random.randint(0, 5) < 2:   # 33%: 랜덤 픽셀
    color = img[ys[idx], xs[idx]]
else:                           # 67%: 평균 색상
    color = np.sum(img[ys,xs], axis=0) / points_num
```

효과:
- **색 대응 유지**: stroke 색 ∈ 실제 머리 색 범위 → 모델이 색 correspondence 학습
- **데이터 증강**: 동일 구조라도 매 iteration 미세하게 다른 색이 주어져 다양성 확보

### 추론 시

```python
# augment=False 시 (추론)
color = np.sum(img[ys,xs], axis=0) / points_num  # 항상 평균 색상
```

사용자가 원하는 색으로 stroke를 그리면:
1. 선택한 색과 가장 가까운 자연스러운 머리 색을 7만개 DB에서 검색
2. 해당 색으로 stroke를 recoloring
3. 모델이 그 색 계열의 머리를 생성

→ 보라색, 파란색 등 비자연 색상도 DB 매핑을 통해 어느 정도 지원 가능

---

## 3. 우리 DiT 방식 (exp1-stroke-color)

### stroke 구분 방법

RGB 값 그대로 양자화하여 stroke를 구분한다.

```python
sketch_q = (sketch_u8 >> self.shift) << self.shift  # 하위 비트 버리기
unique_colors = torch.unique(flat_q, dim=0)          # RGB로 unique stroke 검출
```

GAN과 달리 grayscale 변환 없이 RGB를 직접 사용하므로 stroke collision이 없다.

### 기존 문제 (main 브랜치)

`SketchColorJitter(p=0.8)`: 매 iteration마다 stroke 색을 **완전 랜덤 HSV 색**으로 교체했다.

```
같은 갈색 머리 target → stroke 색이 초록, 파랑, 노랑 등 완전 무관한 색으로 바뀜
```

모델 입장에서 stroke 색과 target 색 사이에 아무 패턴이 없음 → **색을 무시하도록 학습됨**.
구조(선 방향, 교차, 땋임 패턴)만 보고 머리를 생성하며, 색은 학습 데이터 평균으로 수렴.

### 변경 방식 (StrokeColorSampler, p=1.0)

GAN과 동일한 비율로 색 교체:

```python
# 67% 평균 색상, 33% 랜덤 픽셀 (GAN color_coding 방식)
if random.randint(0, 5) < 2:
    sampled_color = hair_pixels_valid[:, idx]        # 랜덤 픽셀
else:
    sampled_color = hair_pixels_valid.float().mean(dim=1)  # 평균 색상
```

`AppearanceJitter`도 제거: target 색을 흔들면 stroke ↔ target 색 대응이 깨지기 때문.

### 추론 시 (미구현, phase1 학습 완료 후 추가 예정)

현재 `infer_custom.py`는 색 교체 없이 레이블 색 그대로 모델에 입력한다.
→ 학습/추론 간 distribution mismatch 발생.

추론 시에는 GAN과 동일하게 **평균 색상**으로 교체 후 입력해야 한다:
```
사용자 stroke (원하는 머리 색) → stroke 영역 평균 색상 계산 → 모델 입력
```

---

## 4. 한계 및 개선 계획

**색 DB 부재**
SketchHairSalon은 7만개 실제 머리 색 DB를 통해 사용자가 선택한 임의 색을 자연스러운 머리 색으로 매핑한다.
우리는 이 DB가 없기 때문에, **학습 데이터에 존재하는 자연 머리 색 범위 안에서만** 색 제어가 가능하다.

```
갈색 stroke → 갈색 머리  ✓ (학습 데이터에 존재)
보라색 stroke → ???       ✗ (학습 중 본 적 없는 색)
```

**개선 계획**
학습 데이터에서 stroke별 평균 색을 추출하여 자연 머리 색 DB를 구축하고, 추론 시 사용자 stroke 색을 nearest neighbor로 매핑하는 방식으로 보완 가능.

**학습 데이터 색 다양성에 의존**
데이터셋에 특정 머리 색이 부족하면 그 색 제어가 약해진다.
색 제어 품질은 데이터셋 내 머리 색 분포에 직접적으로 영향을 받는다.
