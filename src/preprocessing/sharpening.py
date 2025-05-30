import cv2
import numpy as np
from .base import BasePreprocessor

class SharpeningPreprocessor(BasePreprocessor):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening to the image"""
        # Convert to grayscale if the image is colored
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), self.sigma)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # Convert back to original format if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
        return unsharp_mask 