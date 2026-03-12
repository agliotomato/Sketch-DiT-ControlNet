# 교수님 미팅 핵심 정리 — 방향 전환안 (Claude 전달용)

## 0. 결론부터
교수님이 제안하신 핵심은 다음과 같다.

**기존의 “전체 이미지를 직접 생성하는 문제”를 버리고,**
**“머리 영역만 생성하는 문제”로 축소하자.**

즉 연구 문제를 다음처럼 재정의한다.

```text
기존:
(sketch + face + matte) → full image

변경:
(sketch + matte) → hair region image
```

여기서 생성되는 것은 **전체 얼굴 이미지가 아니라, 머리 영역만 있는 이미지**다.  
우리가 대화에서 편의상 이것을 **hair patch / hair region image**라고 부른다.

---

# 1. 교수님이 왜 틀을 바꾸자고 하셨는가

## 1-1. 기존 접근의 문제
기존 접근은 사실상 다음 문제를 한 번에 풀려고 하는 것이었다.

- 얼굴 보존
- 피부/이마/귀 경계 처리
- 머리 구조 생성
- 질감/조명/색감 생성
- 전체 이미지 자연스러움 확보

즉 모델이 한 번에 너무 많은 것을 해야 한다.

현재 데이터 규모는 대략:
- braid: 1000
- unbraid: 3000

이 정도 규모에서 위 문제를 정면으로 풀면,
모델은 스케치를 강하게 따르기보다 **그럴듯한 일반 머리**로 도망갈 가능성이 높다.

교수님 의도는 명확하다.

> 문제를 더 작게 쪼개서, 지금 연구 환경에서 실질적으로 증명 가능한 카드부터 만들자.

---

## 1-2. 교수님이 보신 핵심 병목
교수님은 현재 병목을 **“전체 이미지 생성” 자체**로 보신 것이다.

즉 핵심은:
- 지금 필요한 것은 “얼굴까지 포함한 완성 이미지 생성”이 아님
- 지금 필요한 것은 “스케치를 따라가는 머리 구조 생성”임

그래서 연구 초점을 다음으로 이동시키자는 제안이다.

```text
스케치가 실제로 머리 구조를 지배하게 만들 수 있는가?
```

---

# 2. 교수님 제안의 정확한 구조

교수님이 제안하신 파이프라인은 아래처럼 이해하면 된다.

## Step 1. 머리 영역 생성
```text
(sketch + matte) → hair region image
```

입력:
- sketch: 머리 구조 정보
- matte: 머리 생성 영역

출력:
- 머리 영역 이미지 (배경/얼굴은 없음)
- 즉 matte 내부에만 hair texture가 존재하는 이미지

이 단계가 오늘 미팅에서 새로 강조된 핵심이다.

---

## Step 2. 원본 얼굴 이미지에 합성
생성한 머리 영역 이미지를 얼굴 이미지 위에 붙인다.

```text
composite = hair_region * matte + face_image * (1 - matte)
```

이건 1차 합성이고, 우선은 단순 paste 수준으로 생각하면 된다.

중요한 점:
- 여기서 hair region image는 **전체 얼굴 이미지가 아님**
- 머리 부분만 생성한 결과를 얼굴 위에 올리는 것임

---

## Step 3. 자연스러운 blending / refinement
교수님이 “nano banana처럼 빈 곳과 자연스럽게 합성”이라고 하신 부분은 이 단계다.

즉 단순 paste 후,
경계/빈 영역/헤어라인/조명 mismatch를 **diffusion inpainting 또는 refinement**로 다듬는다.

역할:
- 헤어라인 자연화
- 경계 티 제거
- 조명/질감 정합
- “붙인 티” 제거

중요:
- 이 단계는 **머리 구조를 새로 만드는 단계가 아니라**
- **붙인 결과를 자연스럽게 정리하는 단계**다.

---

# 3. 원 논문(HairSalon / SketchHairSalon)과의 연결

교수님 제안은 완전히 새로운 발상이 아니라,
사실상 **원 논문의 구조를 유지하되 generator를 업그레이드하자는 것**으로 이해하면 된다.

원 논문 흐름은 대략 다음과 같다.

```text
sketch
→ hair matte prediction
→ generator (GAN)
→ hair region 생성
→ face image에 붙임
→ inversion / refinement로 자연화
```

즉 원 논문도 **전체 이미지를 처음부터 끝까지 생성하는 게 아니라**,
중간에 머리 영역만 만들고, 그것을 나중에 자연스럽게 녹이는 구조다.

교수님이 바꾸자고 하신 핵심은 여기다.

```text
기존:
GAN generator → hair region 생성

변경:
DiT / diffusion generator → hair region 생성
```

즉 **파이프라인은 유지하고 generator만 바꾸는 방향**이다.

---

# 4. “hair patch / hair region image”가 정확히 무엇인가

이 부분이 혼동 포인트였기 때문에 명확히 정의한다.

## 정의
hair patch / hair region image란:

- 전체 얼굴 이미지가 아님
- 배경과 얼굴은 제거되거나 비어 있음
- matte가 정의한 머리 영역 내부에만 머리 texture가 존재하는 이미지

즉 개념적으로는:

```text
hair_region = generator(sketch, matte)
```

그리고 시각적으로는:

- 검은 배경 위에 머리만 있는 이미지
또는
- RGBA처럼 머리 외부는 투명/무의미한 값인 이미지

핵심은:
> generator가 만드는 출력은 “완성 얼굴 이미지”가 아니라 “머리 부분만 있는 이미지”다.

---

# 5. 왜 이 방향이 맞는가

## 5-1. 문제 규모 축소
기존 full-image generation은 너무 크다.

새 구조는 모델이 배워야 하는 것을 다음으로 제한한다.

- braid/unbraid 구조
- strand geometry
- hair texture
- shading

즉 다음은 제외된다.

- 얼굴 identity 전체 생성
- 피부/배경 전체 생성
- 전체 장면 자연스러움 전부

이렇게 해야 스케치 조건을 진짜로 따르는지 검증 가능하다.

---

## 5-2. 데이터 규모와 맞는다
현재 데이터 규모에서는
**“머리만 생성하는 문제”**는 해볼 만하지만,
**“전체 인물 이미지를 직접 생성하는 문제”**는 과하다.

즉 이건 단순한 구현 선택이 아니라,
**데이터와 문제 난이도를 맞추는 설계 변경**이다.

---

## 5-3. 실패 원인 분리가 가능하다
기존 방식은 결과가 이상하면
- 머리 생성이 문제인지
- 합성이 문제인지
- conditioning이 문제인지
- 전체 diffusion prior가 문제인지

분리가 어렵다.

반면 새 구조는 단계별 검증이 가능하다.

1. hair region 생성이 잘 되는가?
2. paste했을 때 구조가 맞는가?
3. refinement가 자연스러움을 올리는가?

즉 분석이 가능해진다.

---

# 6. unbraid → braid 순서에 대한 판단

현재 데이터는:
- braid: 1000
- unbraid: 3000

여기서 학습 순서는 **unbraid pretrain → braid fine-tune**이 자연스럽다.

이유:
- unbraid는 texture / shading / 일반 hair prior 학습에 유리
- braid는 구조 학습에 중요
- braid는 복잡하므로 나중에 구조를 강하게 fine-tune하는 게 합리적

단, 중요한 전제가 있다.

> braid 단계에서 모델이 unbraid prior에 먹히면 안 된다.

즉 braid fine-tune의 목적은:
- “머리처럼 보이게 하기”가 아니라
- “braid 구조를 따르게 하기”다.

---

# 7. augmentation에 대한 해석

교수님이 augmentation 이야기를 하신 건
단순 rotate/flip 수준을 넘는 의미로 봐야 한다.

핵심은:

> 지금 데이터 규모에서는 모델을 무작정 키우는 것보다,
> 입력-정답 대응 관계를 깨지 않으면서 다양성을 늘려야 한다.

특히 중요한 augmentation 방향:

## 7-1. sketch color randomization
colored sketch의 색은 머리색 정보가 아니라,
**strand separation / structural separation** 역할이다.

따라서 절대색에 모델이 과적합하면 안 된다.

목표:
- color = appearance signal 이 아님
- color = structural cue 가 되게 만들기

---

## 7-2. sketch thickness jitter
얇은 선이 downsampling에서 죽지 않게,
선 두께 변화에 robust해야 함

---

## 7-3. matte boundary perturbation
완벽한 mask에만 맞추지 않고,
경계가 조금 흔들려도 생성이 되도록 해야 함

---

## 7-4. appearance jitter
구조와 appearance를 분리하기 위해,
target hair image의 밝기/대비/색감을 약하게 흔드는 방향이 유효

---

# 8. 교수님이 말한 “nano banana처럼 자연스럽게 합성”의 정확한 의미

이 말은 “처음부터 완벽한 이미지를 생성하라”가 아니다.

정확한 해석은 다음이다.

1. 먼저 머리 영역 이미지를 생성한다.
2. 그것을 얼굴 위에 paste한다.
3. 경계와 빈 영역만 diffusion이 자연스럽게 다듬게 한다.

즉:

```text
hair region 생성
→ 단순 합성
→ 경계/빈 영역 자연화
```

여기서 diffusion의 역할은:
- 구조 생성기보다
- blending / harmonization 엔진에 가깝다.

따라서 refinement는 전체 이미지를 재생성하는 용도가 아니라,
붙인 결과를 자연화하는 용도다.

---

# 9. 지금 시점의 진짜 핵심 연구 질문

오늘 미팅 기준으로 진짜 핵심 연구 질문은 이것이다.

```text
colored sketch와 matte를 조건으로 주었을 때,
DiT가 braid/unbraid hair structure를 실제로 따르는
hair region image를 생성할 수 있는가?
```

이게 1차 증명 목표다.

그리고 그 다음 질문은:

```text
생성된 hair region을 얼굴 위에 붙였을 때,
diffusion refinement가 자연스럽게 녹여줄 수 있는가?
```

즉 연구는 두 단계로 나뉜다.

### 1차 질문
- 구조를 따르는 hair region 생성 가능 여부

### 2차 질문
- 자연스러운 합성 가능 여부

---

# 10. Claude가 구현/설계 시 반드시 지켜야 할 방향

## 해야 할 것
- 문제를 hair generation task로 고정
- `(sketch + matte) → hair region image`를 명확한 supervision task로 설계
- unbraid pretrain 후 braid fine-tune 전략 유지
- braid 단계에서는 구조 fidelity를 우선
- 합성은 먼저 단순 paste, 이후 refinement 분리
- refinement는 재생성이 아니라 자연화 용도로 제한

---

## 하면 안 되는 것
- 다시 full-image generation 문제로 확장
- diffusion refinement가 braid 구조까지 복구해줄 거라고 가정
- braid / unbraid를 아무 전략 없이 섞어서 학습
- augmentation으로 topology correspondence를 깨뜨림
- “그럴듯한 머리”를 “스케치를 따른 머리”와 혼동

---

# 11. 최종 요약

교수님 제안의 본질은 아래 한 줄로 정리된다.

> **전체 이미지를 직접 생성하지 말고,**
> **DiT로 머리 영역 이미지를 먼저 생성한 뒤,**
> **그 결과를 얼굴에 붙이고 diffusion으로 자연스럽게 정리하자.**

즉 최종 파이프라인은:

```text
(sketch + matte)
→ DiT hair region generator
→ generated hair region image
→ face image에 paste
→ diffusion refinement for natural blending
```

이 방향은
- 문제 규모를 줄이고
- 현재 데이터 조건에 맞추며
- 단계별 검증이 가능하고
- 기존 HairSalon pipeline을 발전시키는 형태다.