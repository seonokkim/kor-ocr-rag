from typing import List, Tuple
import numpy as np
from paddleocr import PaddleOCR
from .base import BaseOCRModel

class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR based Korean OCR Model"""
    
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml"):
        """Initializes the PaddleOCR model.
        
        Args:
            use_gpu (bool): Whether to use GPU.
        """
        super().__init__(config_path)
        
        # Initialize PaddleOCR (using basic settings)
        self.ocr = PaddleOCR(
            lang='korean',  # Korean language mode
            use_angle_cls=True  # Text orientation detection
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing specific to PaddleOCR (BGR -> RGB conversion)"""
        # PaddleOCR expects RGB format images as input
        return image[..., ::-1]  # Convert BGR -> RGB
        
    def predict(self, processed_image: np.ndarray) -> List[Tuple[List[List[int]], Tuple[str, float]]]:
        """Performs prediction using PaddleOCR"""
        # Perform text recognition
        result = self.ocr.ocr(processed_image, cls=True)
        return result
        
    def postprocess(self, prediction_result: List[Tuple[List[List[int]], Tuple[str, float]]]) -> List[Tuple[str, List[float]]]:
        """Postprocesses PaddleOCR prediction results (extract text and bounding boxes, convert types)"""
        predictions = []
        if prediction_result is not None:
            for line in prediction_result:
                # Check if each element in line is a list of length 2 (bbox, (text, confidence) format)
                if isinstance(line, list) and len(line) == 2:
                    bbox = line[0]  # Bounding box coordinates
                    text_info = line[1] # (text, confidence) tuple
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        text = str(text_info[0]) # Convert text to string
                        # confidence = float(text_info[1]) if len(text_info) > 1 else 0.0 # Convert confidence to float
                        
                        # Convert bounding box coordinates to a list of floats
                        if isinstance(bbox, list):
                            bbox_list = [[float(p[0]), float(p[1])] for p in bbox] if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in bbox) else []
                            
                            # Convert bounding box to [x1, y1, x2, y2] format and then to a list of floats
                            if bbox_list:
                                x_coords = [p[0] for p in bbox_list]
                                y_coords = [p[1] for p in bbox_list]
                                bbox_x1y1x2y2 = [float(min(x_coords)), float(min(y_coords)), float(max(x_coords)), float(max(y_coords))]
                                
                                predictions.append((text, bbox_x1y1x2y2))
                            # else: # Invalid bbox format
                            #     print(f"Warning: Invalid bbox format from PaddleOCR: {bbox}")
                        # else: # Invalid bbox format
                        #      print(f"Warning: Invalid bbox format from PaddleOCR: {bbox}")
                # else: # Invalid line format
                #      print(f"Warning: Invalid line format from PaddleOCR: {line}")

        return predictions

    # __call__ method is inherited from BaseOCRModel 