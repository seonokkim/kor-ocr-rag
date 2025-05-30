import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_evaluation_config(
    model_name: str,
    preprocessing_steps: List[str],
    use_gpu: bool = True
) -> Dict[str, Any]:
    """평가 설정을 생성합니다."""
    return {
        'model_name': model_name,
        'preprocessing_steps': preprocessing_steps,
        'use_gpu': use_gpu,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

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

def save_evaluation_results(results: Dict[str, Any], config: Dict[str, Any]):
    """평가 결과를 저장합니다."""
    # 결과 파일명 생성 (날짜_모델_전처리_순번.json 형식)
    today = time.strftime('%Y%m%d')
    model_name = config['model_name']
    preprocess_steps = config['preprocessing_steps']
    
    # 전처리 여부와 방식 결정
    if not preprocess_steps:
        preprocess_info = 'no_preprocess'
    else:
        preprocess_info = '_'.join(preprocess_steps)
    
    # 다음 파일 번호 생성
    next_num = get_next_result_number(model_name, preprocess_info)
    
    # 결과 파일 저장
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{today}_{model_name}_{preprocess_info}_{next_num}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'metrics': results['metrics'],
            'predictions': results['predictions']
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {results_file}")

def load_all_results() -> Dict[str, Any]:
    """모든 평가 결과를 로드합니다."""
    all_results = {}
    results_dir = Path('results')
    
    if not results_dir.exists():
        return all_results
    
    # 모든 JSON 파일에서 결과 로드
    for result_file in results_dir.glob('*.json'):
        if result_file.name == 'performance_report.csv':
            continue
            
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            
        # 파일명을 키로 사용
        all_results[result_file.stem] = result_data
    
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
    """성능 보고서를 생성합니다."""
    if not all_results:
        return pd.DataFrame()
    
    # 결과 데이터 준비
    data = []
    for file_name, result in all_results.items():
        config = result['config']
        metrics = result['metrics']
        data.append({
            'Model': config['model_name'],
            'Preprocessing': '_'.join(config['preprocessing_steps']) if config['preprocessing_steps'] else 'no_preprocess',
            'Item Accuracy': metrics.get('item_accuracy', 0),
            'Char Accuracy': metrics.get('char_accuracy', 0),
            'Inference Time': metrics.get('inference_time', 0)
        })
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(data)
    df = df.sort_values(['Model', 'Preprocessing'])
    
    # 결과 저장
    report_file = Path('results') / 'performance_report.csv'
    df.to_csv(report_file, index=False)
    
    return df 