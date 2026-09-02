# 밀도 CNN 분류 성능 실험

실행: `.venv/bin/python train_density.py`

`density_demo.py`의 지도 시각화와 달리, 이 실험은 CNN을 실제로 학습하고 이전 모델과 분류 성능을 비교합니다.

## 모델

1. Original CNN (max pool): 앞 실험의 원본 CNN 체크포인트 평가값을 재사용합니다.
2. Density CNN (max pool): 원본 입력에 5×5와 11×11의 지역 불량률 두 채널을 추가합니다. 풀링은 기존과 같은 MaxPool입니다.
3. Density CNN (avg pool): 2와 동일한 입력 및 구조에서 MaxPool만 AvgPool로 변경합니다.

밀도는 `불량 마스크를 합성곱한 값 / 유효 영역 마스크를 합성곱한 값`입니다. 고정된 all-ones 필터로 먼저 계산하며, 후속 CNN의 필터는 학습합니다. 밀도 채널은 64×64로 리사이즈한 격자에서 계산하므로 원래 물리적 칩 비율과 완전히 같지는 않습니다. 배경과 padding은 분모에서 제외하며 배경 중심의 출력은 0으로 표시합니다. 원본 채널이 배경과 정상 칩을 구분합니다.

## 평가 조건

이전 실험에서 저장한 `results/polar_comparison/split_manifest.csv`를 그대로 사용합니다. 학습·검증·테스트 생산 lot 및 동일 이미지 중복은 이미 분리했습니다. 원본 데이터의 미라벨 샘플은 제외했습니다. 동일 학습률 0.001, Adam, batch 64, seed 42, 최대 20 epoch, 검증 Macro-F1 기준 patience 5이며 최적 검증 가중치를 복원해 테스트합니다. 클래스 가중치와 증강은 사용하지 않습니다.

밀도 채널 추가로 첫 번째 합성곱 파라미터가 576개 증가합니다. 밀도 Max와 밀도 Avg 모델의 파라미터 수는 같습니다. 단일 seed의 탐색적 비교입니다. 이전에 분석했던 테스트를 재사용하므로 완전히 새로운 독립 최종 평가로 해석하지 마세요. 추가 튜닝이나 선택은 검증 지표를 기준으로 해야 합니다.

## 산출물

`results/density_comparison/performance_comparison.png`: 종합 지표와 유형별 Recall.
`summary.csv`, `recall_by_class.csv`: 수치.
모델별 학습 기록, 체크포인트, 혼동행렬, 예측 내역도 함께 저장됩니다.
결과가 이미 있으면 실행은 덮어쓰기 없이 중단됩니다.
