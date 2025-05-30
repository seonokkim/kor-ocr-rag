import cv2
import numpy as np
from .base import BasePreprocessor

class DenoisingPreprocessor(BasePreprocessor):
    def __init__(self, h: int = 10, template_window_size: int = 7, search_window_size: int = 21):
        self.h = h  # 필터 강도
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거를 적용합니다."""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Non-local Means Denoising 적용
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        
        # 원본 형식으로 변환
        if len(image.shape) == 3:
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return denoised 