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
        'model': model_name,
        'preprocessing': preprocessing_steps,
        'hardware': {
            'use_gpu': use_gpu,
            'device': 'cuda' if use_gpu else 'cpu'
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def save_evaluation_results(
    results: Dict[str, Any],
    config: Dict[str, Any],
    results_dir: str = "results"
) -> str:
    """평가 결과를 저장합니다."""
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']
    preprocess_steps = '_'.join(config['preprocessing']) if config['preprocessing'] else 'no_preprocessing'
    filename = f"eval_{model_name}_{preprocess_steps}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 결과 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    return filepath

def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """모든 평가 결과를 로드하여 DataFrame으로 반환합니다."""
    results = []
    
    for file in Path(results_dir).glob("eval_*.json"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            config = data['config']
            metrics = data['results']['metrics']
            
            results.append({
                'timestamp': config['timestamp'],
                'model': config['model'],
                'preprocessing': ','.join(config['preprocessing']) if config['preprocessing'] else 'none',
                'use_gpu': config['hardware']['use_gpu'],
                'accuracy': metrics['accuracy'],
                'char_accuracy': metrics['char_accuracy'],
                'inference_time': metrics['inference_time']
            })
    
    return pd.DataFrame(results)

def plot_performance_comparison(df: pd.DataFrame, metric: str = 'accuracy'):
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

def generate_performance_report(df: pd.DataFrame, output_dir: str = "results"):
    """성능 보고서를 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 통계
    stats = df.groupby(['model', 'preprocessing']).agg({
        'accuracy': ['mean', 'std'],
        'char_accuracy': ['mean', 'std'],
        'inference_time': ['mean', 'std']
    })
    
    # 컬럼 이름 정리 (멀티인덱스 튜플을 문자열로 변환)
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    # 인덱스를 일반 컬럼으로 변환하여 딕셔너리 키가 튜플이 되지 않도록 함
    stats = stats.reset_index()
    
    stats = stats.round(4)
    
    # 그래프 생성
    plt = plot_performance_comparison(df)
    plt.savefig(os.path.join(output_dir, f'performance_comparison_{timestamp}.png'))
    plt.close()
    
    # 보고서 저장
    report = {
        'timestamp': timestamp,
        'summary_statistics': stats.to_dict(),
        'best_performing_combinations': {
            'accuracy': df.loc[df['accuracy'].idxmax()].to_dict(),
            'char_accuracy': df.loc[df['char_accuracy'].idxmax()].to_dict(),
            'fastest_inference': df.loc[df['inference_time'].idxmin()].to_dict()
        }
    }
    
    with open(os.path.join(output_dir, f'performance_report_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report 