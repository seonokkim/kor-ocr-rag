from abc import ABC, abstractmethod
import torch
import yaml
from pathlib import Path

class BaseOCRModel(ABC):
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> str:
        if self.config['hardware']['use_gpu'] and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @abstractmethod
    def preprocess(self, image):
        """Preprocess the input image"""
        pass
    
    @abstractmethod
    def predict(self, image):
        """Perform OCR prediction on the image"""
        pass
    
    @abstractmethod
    def postprocess(self, prediction):
        """Postprocess the model output"""
        pass
    
    def __call__(self, image):
        """Run the complete OCR pipeline"""
        preprocessed = self.preprocess(image)
        prediction = self.predict(preprocessed)
        return self.postprocess(prediction) 