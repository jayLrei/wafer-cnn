# 밀도 비교 결과

이 결과는 학습된 CNN의 판단이나 정확도가 아니라, 고정된 필터와 풀링으로 계산한 특징입니다.

`real_wafer_density.png`: 기존 train 분할에서 none/Scratch/Loc/Random의 공통 원본 크기를 선택하고, 각 유형·크기 안에서 전체 불량률이 중앙 순위인 샘플을 선택했습니다. 라벨별 대표성이나 구분 가능성을 보장하지 않습니다.

5×5 및 11×11 all-ones 합성곱을 불량 마스크와 유효 칩 마스크에 각각 적용한 뒤 나눕니다. 검은 배경과 padding은 분모에 포함하지 않습니다. 풀링도 두 마스크를 각각 평균 풀링하고 나눕니다. 8×8 adaptive pooling 구역은 입력 크기가 나누어떨어지지 않으면 일부 겹칠 수 있습니다.

`pooling_comparison.png`는 원리를 분리해 보여주는 인공 예시입니다. 두 입력 모두 16/64칸이 불량입니다. 고정 4×4 블록에서 Max Pooling은 두 예시의 출력이 같지만 Average Pooling은 다릅니다. 일반적인 학습 CNN의 Max Pooling도 앞선 학습 필터가 추출한 정보를 전달할 수 있으므로 Max Pooling CNN 전체가 밀도를 구분 못한다는 뜻은 아닙니다.

고정 필터로 밀도를 명시적으로 계산할 수 있다는 것과, 자유롭게 학습한 CNN 필터가 실제로 밀도를 사용한다는 것은 다릅니다. 밀도만으로 9개 유형의 분류 성능이 좋아지는지도 별도 학습·검증이 필요합니다.

재현: 프로젝트 .venv Python으로 `density_demo.py` 실행. 숫자는 density_values.csv와 density_maps.npz에 저장했습니다.
