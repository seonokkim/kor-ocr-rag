from typing import List, Tuple
import numpy as np
from paddleocr import PaddleOCR
from .base import BaseOCRModel

class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR 기반 한글 OCR 모델"""
    
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml"):
        """PaddleOCR 모델을 초기화합니다.
        
        Args:
            use_gpu (bool): GPU 사용 여부
        """
        super().__init__(config_path) # BaseOCRModel 초기화 추가
        
        # use_gpu 값에 따라 0 또는 1 설정 (이 로직은 더 이상 PaddleOCR 초기화 인자로 사용되지 않음)
        # gpu_param = 1 if use_gpu else 0
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 텍스트 방향 감지 (DeprecationWarning 발생 가능성 있음)
            lang='korean'       # 한글 모드
            # use_gpu 인자 제거
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """PaddleOCR에 맞는 전처리 (BGR -> RGB 변환)"""
        # PaddleOCR은 RGB 형식의 이미지를 입력으로 받음
        return image[..., ::-1]  # BGR -> RGB 변환
        
    def predict(self, processed_image: np.ndarray) -> List[Tuple[List[List[int]], Tuple[str, float]]]:
        """PaddleOCR 예측 수행"""
        # 텍스트 인식 수행
        result = self.ocr.ocr(processed_image, cls=True)
        return result
        
    def postprocess(self, prediction_result: List[Tuple[List[List[int]], Tuple[str, float]]]) -> List[Tuple[str, List[float]]]:
        """PaddleOCR 예측 결과 후처리 (텍스트와 바운딩 박스 추출)"""
        predictions = []
        if prediction_result is not None:
            for line in prediction_result:
                for word_info in line:
                    bbox = word_info[0]  # 바운딩 박스 좌표
                    text = word_info[1][0]  # (bbox, (text, confidence)) 형식
                    # confidence = word_info[1][1]  # 신뢰도
                    
                    # 바운딩 박스를 [x1, y1, x2, y2] 형식으로 변환
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    bbox_x1y1x2y2 = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    predictions.append((text, bbox_x1y1x2y2))
        
        return predictions

    # __call__ 메서드는 BaseOCRModel에서 상속받아 사용합니다. 