HairCLIP 버전
제 지식 기준(2025.08)으로 HairCLIPv2 (ICCV 2023) 까지가 최신입니다. v3는 공식 확인된 게 없어서, 비교 대상은 HairCLIPv2로 쓰는 게 맞습니다.

Teaser Figure 구성안
2행 × 4열 (행 = 예제, 열 = 모델)

NanoBanana	BrushNet	SHS	Ours
Braid (땋은 머리)	실패 케이스	실패 케이스	무난	가장 좋은 것
Unbraid (정면, 비-브레이드)	실패 케이스	실패 케이스	무난	가장 좋은 것
예제 선정 기준:

best_worst.md의 Braid Best 5 / Overall Best 20에서 NanoBanana·BrushNet이 구조 무너지는 것 선택
스케치 스타일: 금발 그라데이션 계열 (실제 헤어 디자인 스케치 참고), 알록달록 랜덤 색 X
정량 비교 테이블 구성안

SHS | HairCLIPv2 | BrushNet | Ours
메트릭은 기존 eval_all.py 기반:

카테고리	메트릭
Sketch Fidelity	Edge IoU↑, Chamfer Dist↓
Generation Quality	Hair FID↓, LPIPS(GT)↓, SSIM↑
Boundary Quality	Boundary FID↓
Identity	ArcFace Cos↑
색 표현 능력 별도 평가가 필요하면: 알록달록 스트록 결과로 SHS vs Ours 서브 테이블 추가 가능 (HairCLIPv2·BrushNet은 색 제어 방식이 달라서 직접 비교 어려움).

사용자 연구 (User Study) 설계안
자극: 알록달록보다는 자연스러운 스타일 범위 — 내추럴 브라운 → 그라데이션 → 파스텔 펑키 정도
비교 쌍: SHS vs Ours, BrushNet vs Ours 각각 랜덤 순서 제시
질문: "어느 결과가 스케치의 의도에 더 잘 맞습니까?" (forced choice)
샘플 수: braid 15장 + unbraid 15장 = 30쌍 권장
지금 당장 필요한 작업이 뭔지 알려주시면 바로 진행합니다:

BrushNet / NanoBanana 추론 결과가 이미 있는지 → teaser용 예제 선정
eval_all.py에 HairCLIPv2 / BrushNet 추가해서 정량 테이블 생성
색 표현 능력 서브 평가 스크립트