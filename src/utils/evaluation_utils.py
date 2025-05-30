import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np

# Helper function to convert numpy types to standard Python types
def convert_numpy_types(obj):
    """Recursively convert numpy types to standard Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

def create_evaluation_config(
    model_name: str,
    preprocessing_steps: List[str],
    use_gpu: bool
) -> Dict[str, Any]:
    """Creates evaluation configuration.
        
    Args:
        model_name (str): Name of the model being evaluated.
        preprocessing_steps (List[str]): List of preprocessing steps applied.
        use_gpu (bool): Whether GPU was used.
        
    Returns:
        Dict[str, Any]: Evaluation configuration dictionary.
    """
    # Generate current timestamp (YYYYMMDD) for filename
    timestamp = time.strftime("%Y%m%d")
    
    # Generate filename based on model and preprocessing steps
    preprocess_name = "_".join(preprocessing_steps) if preprocessing_steps else "no_preprocess"
    
    # Combine model name and preprocessing name for config name
    config_name = f"{timestamp}_{model_name}_{preprocess_name}"
    
    # Create config dictionary
    config = {
        'config_name': config_name,
        'model_name': model_name,
        'preprocessing_steps': preprocessing_steps,
        'use_gpu': use_gpu,
        'timestamp': timestamp
    }
    
    return config

def get_next_result_number(model_name: str, preprocess_info: str) -> int:
    """Generates the next result file number for a given model and preprocessing combo.
        
    Args:
        model_name (str): Name of the model.
        preprocess_info (str): Preprocessing information string.
        
    Returns:
        int: The next sequential file number.
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find files from today with the same model and preprocessing info
    today = time.strftime('%Y%m%d')
    pattern = f"{today}_{model_name}_{preprocess_info}_*.json"
    existing_files = list(results_dir.glob(pattern))
    
    if not existing_files:
        return 1
    
    # Find the highest number
    numbers = [int(f.stem.split('_')[-1]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def get_next_report_number() -> int:
    """Generates the next performance report file number.
        
    Returns:
        int: The next sequential file number.
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find report files from today
    today = time.strftime('%Y%m%d')
    pattern = f"{today}_performance_report_*.csv"
    existing_files = list(results_dir.glob(pattern))
    
    if not existing_files:
        return 1
    
    # Find the highest number
    numbers = [int(f.stem.split('_')[-1]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def save_evaluation_results(results: Dict[str, Any], eval_config: Dict[str, Any]):
    """Saves evaluation results to a JSON file.
        
    Args:
        results (Dict[str, Any]): Dictionary containing evaluation results.
        eval_config (Dict[str, Any]): Evaluation configuration dictionary.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename: timestamp_modelname_preprocessingcombo_sequence.json
    preprocess_tag = "_".join(eval_config['preprocessing_steps']) if eval_config['preprocessing_steps'] else "no_preprocess"
    today = time.strftime('%Y%m%d')
    next_num = get_next_result_number(eval_config['model_name'], preprocess_tag)
    filename = f"{today}_{eval_config['model_name']}_{preprocess_tag}_{next_num}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert numpy types in metrics and predictions before saving
    serializable_metrics = convert_numpy_types(results.get('metrics', {}))
    serializable_predictions = convert_numpy_types(results.get('predictions', []))
    
    serializable_results = {
        'config': eval_config,
        'metrics': serializable_metrics,
        'predictions': serializable_predictions
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation results saved to {filepath}")

def load_all_results() -> Dict[str, Any]:
    """Loads all evaluation results from the results directory.
        
    Returns:
        Dict[str, Any]: Dictionary containing all loaded results.
    """
    all_results = {}
    results_dir = "results"
    if not os.path.exists(results_dir):
        return all_results
    
    # Search for all JSON files (currently not considering subfolders)
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
                # Extract config information from filename (e.g., 20231027_base_easyocr_sharpening_1.json)
                filename = os.path.basename(filepath)
                parts = filename.replace('.json', '').split('_')
                
                # Using filename as key for easy management and to prevent duplicates
                config_key = filename
                
                # Separate metrics information from config information
                metrics = result_data.get('metrics', {})
                
                # Use config information stored within the file, fall back to filename parsing if needed
                eval_config_from_file = {
                     'config_name': result_data.get('config', {}).get('config_name', filename),
                     'model_name': result_data.get('config', {}).get('model_name', parts[1] if len(parts) > 1 else 'unknown'),
                     'preprocessing_steps': result_data.get('config', {}).get('preprocessing_steps', parts[2:-1] if len(parts) > 3 else []), # Estimate from filename
                     'use_gpu': result_data.get('config', {}).get('use_gpu', False), # Default value
                     'timestamp': result_data.get('config', {}).get('timestamp', parts[0] if len(parts) > 0 else '')
                }
                
                all_results[config_key] = {
                    'config': eval_config_from_file,
                    'metrics': metrics
                }
                
        except Exception as e:
            print(f"Warning: Failed to load results from {filepath}: {e}")
            
    return all_results

def plot_performance_comparison(df: pd.DataFrame, metric: str = 'item_accuracy'):
    """성능 비교 그래프를 생성합니다."""
    plt.figure(figsize=(12, 6))
    
    # 모델별 성능 비교
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='model', y=metric)
    plt.title(f'Model Performance Comparison ({metric})')
    plt.xticks(rotation=45)
    
    # 전처리 효과 비교
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='preprocessing', y=metric)
    plt.title(f'Preprocessing Effect on {metric}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def generate_performance_report(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Analyzes evaluation results, generates a performance report, and saves it as CSV.
        
    Args:
        all_results (Dict[str, Any]): Dictionary containing all loaded evaluation results.
        
    Returns:
        pd.DataFrame: Pandas DataFrame containing the performance report.
    """
    report_list = []
    
    for config_key, result_data in all_results.items():
        config = result_data['config']
        metrics = result_data['metrics']
        
        row = {
            'model_name': config['model_name'],
            'preprocessing_steps': "_".join(config['preprocessing_steps']) if config['preprocessing_steps'] else 'no_preprocessing',
            'item_accuracy': metrics.get('item_accuracy', 0),
            'char_accuracy': metrics.get('char_accuracy', 0),
            'inference_time': metrics.get('inference_time', 0),
        }
        
        # Add detailed metrics
        for metric_type in ['type', 'region', 'length', 'size']:
            metric_key = f'{metric_type}_accuracies'
            if metric_key in metrics:
                for k, v in metrics[metric_key].items():
                    row[f'average_{metric_type}_{k}'] = v

        # Add Text Similarity metrics
        if 'text_similarity' in metrics:
            ts_metrics = metrics['text_similarity']
            row['average_normalized_levenshtein'] = ts_metrics.get('normalized_levenshtein', 0)
            row['average_bleu_score'] = ts_metrics.get('bleu_score', 0)
            if 'rouge_scores' in ts_metrics:
                 for r_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                      row[f'average_rouge_{r_metric}'] = ts_metrics['rouge_scores'].get(r_metric, 0)
        
        report_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(report_list)
    
    # Save to CSV file
    results_dir = 'results'
    today = time.strftime('%Y%m%d')
    next_num = get_next_report_number()
    report_filepath = os.path.join(results_dir, f"{today}_performance_report_{next_num}.csv")
    df.to_csv(report_filepath, index=False, encoding='utf-8')
    
    print(f"Performance report saved to {report_filepath}")

    return df 