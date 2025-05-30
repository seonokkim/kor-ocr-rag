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
    use_gpu: bool = True
) -> Dict[str, Any]:
    """평가 설정을 생성합니다."""
    # 현재 타임스탬프 (년월일시분초) 및 고유 식별자 생성
    timestamp = time.strftime("%Y%m%d") # 일별 폴더 대신 파일명에 포함
    # uuid = uuid.uuid4().hex[:6] # 짧은 고유 식별자
    
    # 전처리 단계 이름을 포함한 파일 이름 생성
    preprocess_name = "_".join(preprocessing_steps) if preprocessing_steps else "no_preprocess"
    
    # 모델 이름과 전처리 이름을 조합하여 config 이름 생성
    config_name = f"{timestamp}_{model_name}_{preprocess_name}"
    
    # 설정 딕셔너리 생성
    config = {
        'config_name': config_name,
        'model_name': model_name,
        'preprocessing_steps': preprocessing_steps,
        'use_gpu': use_gpu,
        'timestamp': timestamp
    }
    
    return config

def get_next_result_number(model_name: str, preprocess_info: str) -> int:
    """다음 결과 파일 번호를 생성합니다."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # 오늘 날짜의 파일 찾기
    today = time.strftime('%Y%m%d')
    pattern = f"{today}_{model_name}_{preprocess_info}_*.json"
    existing_files = list(results_dir.glob(pattern))
    
    if not existing_files:
        return 1
    
    # 가장 큰 번호 찾기
    numbers = [int(f.stem.split('_')[-1]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def save_evaluation_results(results: Dict[str, Any], eval_config: Dict[str, Any]):
    """평가 결과를 JSON 파일로 저장합니다."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 파일 이름 생성: 모델이름_전처리조합_타임스탬프.json
    preprocess_tag = "_".join(eval_config['preprocessing_steps']) if eval_config['preprocessing_steps'] else "no_preprocess"
    filename = f"{eval_config['timestamp']}_{eval_config['model_name']}_{preprocess_tag}_1.json" # _1은 추후 여러번 실행시 인덱스 추가 고려
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
    """results 디렉토리의 모든 평가 결과를 로드합니다."""
    all_results = {}
    results_dir = "results"
    if not os.path.exists(results_dir):
        return all_results
    
    # 모든 JSON 파일 검색 (하위 폴더는 현재 고려 안함)
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
                # 파일 이름에서 설정 정보 추출 (예: 20231027_base_easyocr_sharpening_1.json)
                filename = os.path.basename(filepath)
                parts = filename.replace('.json', '').split('_')
                
                # 파일명 파싱 로직 개선 (타임스탬프_모델명_전처리조합) 필요
                # 여기서는 간단히 filename을 키로 사용
                
                # 설정 정보와 결과 분리
                # filename 자체를 키로 사용하여 중복 방지 및 관리를 쉽게 함
                config_key = filename
                
                # 메트릭 정보만 저장하고 설정 정보는 분리
                metrics = result_data.get('metrics', {})
                
                # 설정 정보 (파일명에서 재구성 또는 파일 내부에 저장된 정보 사용)
                # 파일 내부에 저장된 설정 정보가 더 정확하므로 그것을 사용
                eval_config_from_file = {
                     'config_name': result_data.get('config', {}).get('config_name', filename),
                     'model_name': result_data.get('config', {}).get('model_name', parts[1] if len(parts) > 1 else 'unknown'),
                     'preprocessing_steps': result_data.get('config', {}).get('preprocessing_steps', parts[2:-1] if len(parts) > 3 else []), # 파일명에서 추정
                     'use_gpu': result_data.get('config', {}).get('use_gpu', False), # 기본값
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

def generate_performance_report(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """평가 결과를 분석하여 성능 보고서를 생성하고 CSV로 저장합니다."""
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
        
        # 상세 메트릭 추가
        for metric_type in ['type', 'region', 'length', 'size']:
            metric_key = f'{metric_type}_accuracies'
            if metric_key in metrics:
                for k, v in metrics[metric_key].items():
                    row[f'average_{metric_type}_{k}'] = v

        # Text Similarity 메트릭 추가
        if 'text_similarity' in metrics:
            ts_metrics = metrics['text_similarity']
            row['average_normalized_levenshtein'] = ts_metrics.get('normalized_levenshtein', 0)
            row['average_bleu_score'] = ts_metrics.get('bleu_score', 0)
            if 'rouge_scores' in ts_metrics:
                 for r_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                      row[f'average_rouge_{r_metric}'] = ts_metrics['rouge_scores'].get(r_metric, 0)
        
        report_list.append(row)
    
    # DataFrame 생성
    df = pd.DataFrame(report_list)
    
    # CSV 파일로 저장
    report_filepath = os.path.join('results', 'performance_report.csv')
    df.to_csv(report_filepath, index=False, encoding='utf-8')
    
    print(f"Performance report saved to {report_filepath}")

    # 기존 딕셔너리 형식 보고서도 반환 (필요시)
    # 여기서는 간단히 DataFrame을 반환하거나 None을 반환하도록 수정
    return df # 또는 None 