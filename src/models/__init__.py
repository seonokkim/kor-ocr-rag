from .base import BaseOCRModel
from .easyocr import EasyOCRModel

__all__ = [
    'BaseOCRModel',
    'EasyOCRModel',
]

# PaddleOCR 모델은 선택적으로 로드
try:
    from .paddleocr import PaddleOCRModel
    __all__.append('PaddleOCRModel')
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"\nWarning: PaddleOCR 모델을 불러올 수 없습니다 - {str(e)}")
    print("PaddleOCR 모델은 평가에서 제외됩니다.")
    PADDLEOCR_AVAILABLE = False

# 모델 로더 함수 등 필요한 유틸리티 임포트
# from .model_loader import load_model
