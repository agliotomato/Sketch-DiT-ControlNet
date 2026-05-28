### Stage 1 (완료) 변경 없음



DiT frozen + Hair ControlNet 학습 + Matte-Gated Schedule
v2 inference compositor로 평가 (9/11 우세 with combined 573)


### Stage 2 (... 진행 중) — 변경 없음
Inverse LoRA + FPN decoder + 외부 matte + edge loss + color sampling 3×3
Cycle loss는 Stage 2에서 안 함 (warm-start)

 
### Stage 3  — 재설계 필요
원래 계획 ("alternating + LoRA 공유/분리")은 의미 없음. 대신 두 가지 옵션:

#### Option A (Conservative)
 Inverse-only refinement with cycle-ControlNet은 Stage 1 그대로 frozen-Inverse LoRA + FPN decoder를 cycle loss로 refine-"bidirectional"이 약함 — Forward는 inverse 결과 평가만
안전, 빠름

#### Option B (Aggressive): Joint cycle finetuning
ControlNet과 Inverse module 모두 학습 가능 상태로 풀고
Cycle loss로 서로 join training
Mini-batch alternating: forward batch는 ControlNet update, inverse batch는 Inverse module update + cycle 양방향
"Cycle-consistent bidirectional joint training" claim 가능
Forward 회귀 위험 ↑ → lr decoupling 필요

### Things to do
우리가 논문 작업 전에 
"Matte-Gated Residual Schedule" 이것의 결과가 잘 나와서 진행하려고 한 것과 관련
시그래프 수준에서는 단독 디자인만으로 강행하는 것이 현실적으로 어려움 (그래서 bidirectional cyclic 트렌드 적용 - 스케치 가이드와 연동. 모티베이션 충분) 
만약 단독으로 한다고 하더라도, Gated Residual 의 정당성을 위해선 다음의 ablation study 가 필요

### List up
먼저, 위의 stage 2 진행하시고 결과를 본 다음
(외부 matte 사용 + 기존 forward LoRA 실수?! fix)
결과 보고 stage 3 을 바로 진행합시다.
Gated Residual 의 정당성을 위해선 다음의 ablation study 
==> 이것은 위의 실험이 완료되면 반드시 실험 진행할 것으로 리스트업 해 두세요

### 0514
inverse로 뽑은 sketch는 기존 sketch 형태가 아님
기존 sketch 데이터자체가 강력한 brush로 그린거 -> 이 스타일로 나오면 좋음
벡터형 라인으로 바꾸어주는 tool 이 있을거임. svg. 자동으로 스케치에서 썻던 픽셀 이미지.
벡터형으로 잡아주는 게 있을거임. ux 측면에서 더 좋아질 수 있,ㅇㅁ
걔를 가지고 tuning 된 것이 forward로 들어갈 수 있도록
input paint로 들어가게 하는것에는 한계가 있을거임

cyclic으로 갔을 때, forward checkpoint vs cyclic 으로 업데이트
cyclic으로 갔을 때, 더 좋아진다면 강한 claim.
forward 로 갔을때 좋으면 좋음. 정성적으로 갔을 때 좋으면 좋음