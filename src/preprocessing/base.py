from abc import ABC, abstractmethod
import cv2
import numpy as np

class BasePreprocessor(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the input image"""
        pass
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run the preprocessing pipeline"""
        return self.process(image) 