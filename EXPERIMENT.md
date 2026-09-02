# 웨이퍼 원본 / 극좌표 / 결합 CNN 비교

기존 두 노트북은 그대로 보존합니다. `run_experiment.py`가 새 실험입니다.

## 실행

프로젝트 폴더에서:

```sh
.venv/bin/python run_experiment.py --data LSWMD.pkl --output results/polar_comparison
```

새 환경에서는 `python -m pip install -r requirements-polar.txt`로 의존성을 설치합니다.
기존 결과 폴더가 있으면 덮어쓰지 않으므로 다른 `--output`을 지정하세요.

## 비교 방법

- Cartesian: 제공된 기본 CNN과 같은 Conv 32/64, FC 64/9 구조, 원본 64×64 입력.
- Polar: 같은 구조에 극좌표 입력. 가로는 각도(오른쪽=0°, 반시계), 세로는 중심부터 바깥 거리. 최근접 좌표로 0/1/2를 유지하며 보간 평균을 사용하지 않습니다.
- Fusion: 원본과 극좌표의 별도 CNN에서 얻은 64개 특징을 결합해 분류. 파라미터 수가 더 많으므로 순수한 전처리 효과 비교는 Cartesian 대 Polar로 합니다.
- 손실함수(CrossEntropy), Adam(0.001), batch 64 및 분할은 공통입니다. 이번 실험에는 클래스 가중치나 증강을 섞지 않습니다.
- 최대 20 epoch, 검증 Macro-F1이 5 epoch 동안 개선되지 않으면 종료하고 최고 가중치를 복원합니다.
- 테스트는 각 모델의 최적 검증 가중치를 선택한 뒤 평가합니다. 여러 테스트 결과를 보고 추가 튜닝하면 독립 최종 평가가 아니므로 이후 튜닝은 검증 지표로만 합니다.

## 데이터 처리와 해석

라벨 없는 데이터는 제외하고 실제 `none` 라벨만 정상 패턴 클래스로 인정합니다. 정상은 최대 5,000개 사용합니다. 동일 맵 중복은 제거하고 서로 다른 라벨이 붙은 동일 맵은 제외합니다. 원본 데이터 파일은 수정하지 않습니다.
기본 분할은 lotName을 묶어 약 60/20/20으로 나누므로 생산 묶음이 학습·검증·테스트에 겹치지 않습니다. 클래스별 샘플 수는 `class_counts.csv`에서 확인하세요. 원래 노트북의 무작위 80/20 분할과 수치를 직접 비교하면 안 됩니다. `--split stratified`는 대안이지만 생산 묶음이 겹칠 수 있습니다.

좌표 변환은 칩이 있는 영역의 중심과 반지름을 추정합니다. 타원처럼 보이는 맵이나 원본 해상도 차이, 중심 부근의 반복 샘플링, 극좌표 0°/360° 경계는 성능에 영향을 줄 수 있습니다. 원본 정보 손실을 줄이려고 두 표현 모두 원래 맵에서 직접 계산합니다.
단일 seed 실험이므로 결과는 탐색적 비교입니다. 소수 클래스의 Recall은 테스트 샘플 수와 함께 읽으세요. 정상 5,000개로 조절된 평가이므로 실제 공장 데이터 비율의 정확도를 나타내지 않습니다.

## 산출물

- `recall_comparison.png`: 실제 테스트의 클래스별 Recall 비교 그래프
- `representations.png`: 학습 샘플의 원본/극좌표 비교
- `summary.csv`, `recall_by_class.csv`: 종합 지표와 Recall/지원 샘플 수
- `split_manifest.csv`, `class_counts.csv`, `config.json`: 분할과 실행 설정
- 모델별 `*_history.csv`, `*_report.json`, `*_confusion.csv`, `*_predictions.csv`, `*_best.pt`

Recall = 해당 유형으로 올바르게 예측한 수 / 실제 해당 유형 수.
