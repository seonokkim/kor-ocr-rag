import numpy as np
from typing import List, Dict
import pandas as pd
from datetime import datetime
import json
import os

class OCREvaluator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Calculate various OCR evaluation metrics"""
        metrics = {}
        
        # Calculate accuracy (exact match)
        exact_matches = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        metrics['accuracy'] = exact_matches / len(predictions) if predictions else 0
        
        # Calculate character-level metrics
        total_chars_pred = sum(len(p) for p in predictions)
        total_chars_true = sum(len(g) for g in ground_truth)
        correct_chars = sum(sum(1 for c1, c2 in zip(p, g) if c1 == c2) 
                          for p, g in zip(predictions, ground_truth))
        
        metrics['char_accuracy'] = correct_chars / total_chars_true if total_chars_true > 0 else 0
        
        return metrics
    
    def save_results(self, 
                    metrics: Dict, 
                    config: Dict, 
                    model_name: str, 
                    preprocessing_steps: List[str]) -> str:
        """Save evaluation results with timestamp and configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            'timestamp': timestamp,
            'model': model_name,
            'preprocessing': preprocessing_steps,
            'metrics': metrics,
            'config': config
        }
        
        filename = f"results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        return filepath
    
    def compare_results(self) -> pd.DataFrame:
        """Load and compare all results"""
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.results_dir, filename), 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append({
                        'timestamp': result['timestamp'],
                        'model': result['model'],
                        'preprocessing': ','.join(result['preprocessing']),
                        'accuracy': result['metrics']['accuracy'],
                        'char_accuracy': result['metrics']['char_accuracy']
                    })
        
        return pd.DataFrame(results) 