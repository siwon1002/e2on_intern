# e2on_intern — 낙상 탐지 모델 (YOLO26s)

산업현장 CCTV 영상 기반 낙상(Falldown) 탐지 객체탐지 모델 학습 산출물입니다.
비교 실험 로그와 시행착오 기록은 '성능평가보고서' pdf 문서에 정리되어 있습니다.

## 데이터셋

- 클래스: 2개 (`Normal`, `Falldown`)
- 설정 파일: `configs/falldown_2class.yaml`
- 데이터셋 이미지·라벨은 본 저장소에 포함되어 있지 않습니다. 사용 시 yaml의 경로를 직접 수정해야 합니다.

## 실험 명명 규칙

실험 폴더명은 Ultralytics가 자동 생성한 실행명(`train-N`)을 그대로 사용합니다.
각 실행명이 어떤 변인을 조정한 실험인지는 아래 표를 참고하세요.

| 실행명 | 조정한 변인 | 비고 |
|---|---|---|
| `train-10` | `weight_decay=0.001` | 최종 채택 모델 |

본 저장소에는 최종 채택 모델만 포함되어 있으며, 그 외 비교 실험은 노션 문서를 참고하세요.

## 최종 모델 (`runs/detect/train-10`)

`yolo26s.pt` 사전학습 가중치에서 출발해 `weight_decay`를 단일 변인으로 조정한 실험입니다.

| 항목 | 값 |
|---|---|
| 모델 | yolo26s |
| imgsz | 1280 |
| weight_decay | 0.001 |
| epochs | 100 |
| batch | 16 |
| optimizer | auto |

성능 (`results.csv` 최종 에폭 기준)

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.9656 | 0.9153 | 0.9438 | 0.9381 |

위 수치는 전체 클래스 평균값입니다. 클래스별 지표가 필요한 경우 `model.val()`을 다시 실행해 확인하세요.

## 폴더 구조

```
configs/                        데이터셋 설정 yaml
notebooks/                      학습 및 검증 노트북
runs/detect/train-10/           학습 결과 (args.yaml, results.csv, 그래프)
runs/detect/train-10/weights/   최종 모델 가중치 (best.pt)
```

`runs/detect/` 경로는 Ultralytics 기본 출력 경로 규칙을 따릅니다.

## 재현 방법

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")          # 사전학습 가중치는 자동 다운로드됩니다
model.train(
    data="configs/falldown_2class.yaml",
    imgsz=1280,
    weight_decay=0.001,
)
```

학습 환경: Ultralytics 8.4.93 / Python 3.12 / NVIDIA H100 80GB

## 참고 사항

- 사전학습 가중치(`yolo26s.pt`)는 Ultralytics에서 공개 배포하는 파일로, 실행 시 자동 다운로드되므로 저장소에 포함하지 않았습니다. 인터넷이 차단된 환경에서 재현할 경우 별도로 준비해야 합니다.
- 마지막 에폭 체크포인트(`last.pt`)는 `best.pt`와 중복되어 제외했습니다.
- `args.yaml`은 학습 당시 Ultralytics가 자동 생성한 기록이므로 수정하지 않았습니다. 내부 `save_dir` 항목은 학습 서버의 원래 절대경로를 가리킵니다.
