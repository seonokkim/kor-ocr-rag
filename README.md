# Korean OCR Evaluation Framework

한글 OCR 성능 평가를 위한 종합적인 프레임워크입니다.

## 주요 기능

- 다양한 OCR 모델 지원
  - EasyOCR
  - PaddleOCR
  - Tesseract
  - YOLO 기반 OCR

- 이미지 전처리 모듈
  - 샤프닝
  - 노이즈 제거
  - 이진화

- GPU/CPU 선택 가능
- 다양한 평가 지표 제공
- 실험 결과 저장 및 비교 분석

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

1. 설정 파일 수정 (`configs/default_config.yaml`)
   - 사용할 모델 선택
   - 전처리 방법 선택
   - GPU 사용 여부 설정

2. OCR 실행
```python
from src.models import EasyOCRModel
from src.preprocessing import SharpeningPreprocessor

# 모델 초기화
model = EasyOCRModel()

# 전처리기 초기화
preprocessor = SharpeningPreprocessor()

# 이미지 처리
result = model(preprocessor(image))
```

3. 성능 평가
```python
from src.evaluation import OCREvaluator

evaluator = OCREvaluator()
metrics = evaluator.calculate_metrics(predictions, ground_truth)
evaluator.save_results(metrics, config, model_name, preprocessing_steps)
```

## 프로젝트 구조

```
kor-ocr-rag/
├── data/                      # 데이터 폴더
├── src/
│   ├── models/               # OCR 모델 구현
│   ├── preprocessing/        # 이미지 전처리 모듈
│   ├── evaluation/          # 성능 평가 모듈
│   └── utils/               # 유틸리티 함수
├── tests/                   # 테스트 코드
├── configs/                 # 설정 파일
├── results/                 # 실험 결과 저장
└── requirements.txt         # 의존성 패키지
```

## 라이선스

MIT License 