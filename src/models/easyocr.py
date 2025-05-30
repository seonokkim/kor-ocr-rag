import easyocr
import numpy as np
from .base import BaseOCRModel

class EasyOCRModel(BaseOCRModel):
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        super().__init__(config_path)
        self.reader = easyocr.Reader(['ko'], gpu=self.device == "cuda")
    
    def preprocess(self, image):
        """Basic preprocessing for EasyOCR"""
        if isinstance(image, str):
            return image
        return image
    
    def predict(self, image):
        """Perform OCR using EasyOCR"""
        results = self.reader.readtext(image)
        return results
    
    def postprocess(self, prediction):
        """Extract text from EasyOCR results"""
        return [text for _, text, _ in prediction] 