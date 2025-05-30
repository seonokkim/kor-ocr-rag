from typing import List, Tuple
import numpy as np
import easyocr
from .base import BaseOCRModel

class EasyOCRModel(BaseOCRModel):
    """EasyOCR based Korean OCR Model"""
    
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml"):
        """Initializes the EasyOCR model.
        
        Args:
            use_gpu (bool): Whether to use GPU.
        """
        super().__init__(config_path) # Initialize BaseOCRModel
        self.reader = easyocr.Reader(
            ['ko'],  # Korean language mode
            gpu=use_gpu
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing specific to EasyOCR (add if needed)"""
        # EasyOCR processes BGR images directly, no additional preprocessing here
        return image
    
    def predict(self, processed_image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """Performs prediction using EasyOCR"""
        # Perform text recognition
        results = self.reader.readtext(processed_image)
        return results
        
    def postprocess(self, prediction_result: List[Tuple[List[List[int]], str, float]]) -> List[Tuple[str, List[float]]]:
        """Postprocesses EasyOCR prediction results (extract text and bounding boxes, convert types)"""
        predictions = []
        if prediction_result is not None:
            for (bbox, text, prob) in prediction_result:
                # Convert bounding box coordinates to a list of floats
                bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                
                # Convert bounding box to [x1, y1, x2, y2] format and then to a list of floats
                x_coords = [p[0] for p in bbox_list]
                y_coords = [p[1] for p in bbox_list]
                bbox_x1y1x2y2 = [float(min(x_coords)), float(min(y_coords)), float(max(x_coords)), float(max(y_coords))]
                
                predictions.append((text, bbox_x1y1x2y2))
        
        return predictions

    # __call__ method is inherited from BaseOCRModel 