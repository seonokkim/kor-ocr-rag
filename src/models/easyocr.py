from typing import List, Tuple
import numpy as np
import easyocr
from .base import BaseOCRModel

class EasyOCRModel(BaseOCRModel):
    """EasyOCR 기반 한글 OCR 모델"""
    
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml"):
        """EasyOCR 모델을 초기화합니다.
        
        Args:
            use_gpu (bool): GPU 사용 여부
        """
        super().__init__(config_path) # BaseOCRModel 초기화 추가
        self.reader = easyocr.Reader(
            ['ko'],  # 한글 모드
            gpu=use_gpu
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """EasyOCR에 맞는 전처리 (필요시 추가)"""
        # EasyOCR은 BGR 이미지를 직접 처리하므로 추가 전처리 없음
        return image
    
    def predict(self, processed_image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """EasyOCR 예측 수행"""
        # 텍스트 인식 수행
        results = self.reader.readtext(processed_image)
        return results
        
    def postprocess(self, prediction_result: List[Tuple[List[List[int]], str, float]]) -> List[Tuple[str, List[float]]]:
        """EasyOCR 예측 결과 후처리 (텍스트와 바운딩 박스 추출 및 타입 변환)"""
        predictions = []
        if prediction_result is not None:
            for (bbox, text, prob) in prediction_result:
                # 바운딩 박스 좌표를 float 리스트로 변환
                bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                
                # 바운딩 박스를 [x1, y1, x2, y2] 형식으로 변환 후 float 리스트로 변환
                x_coords = [p[0] for p in bbox_list]
                y_coords = [p[1] for p in bbox_list]
                bbox_x1y1x2y2 = [float(min(x_coords)), float(min(y_coords)), float(max(x_coords)), float(max(y_coords))]
                
                predictions.append((text, bbox_x1y1x2y2))
        
        return predictions

    # __call__ 메서드는 BaseOCRModel에서 상속받아 사용합니다. 