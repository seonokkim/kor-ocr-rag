import cv2
import numpy as np
from .base import BasePreprocessor

class BinarizationPreprocessor(BasePreprocessor):
    def __init__(self, method: str = 'adaptive', block_size: int = 11, c: int = 2):
        self.method = method  # 'adaptive' or 'otsu'
        self.block_size = block_size
        self.c = c
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """이진화를 적용합니다."""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 이진화 방법 선택
        if self.method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                self.c
            )
        else:  # otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 원본 형식으로 변환
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary 