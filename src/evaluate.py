import os
import time
import yaml
from typing import List, Dict, Any
import cv2
import numpy as np
from pathlib import Path

from models import EasyOCRModel
from preprocessing import SharpeningPreprocessor
from utils.evaluation_utils import (
    create_evaluation_config,
    save_evaluation_results,
    load_all_results,
    generate_performance_report
)

def load_config(config_path: str = "configs/default_config.yaml") -> Dict:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_images(data_dir: str) -> List[np.ndarray]:
    """테스트 이미지를 로드합니다."""
    images = []
    for img_path in Path(data_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    return images

def evaluate_model(
    model,
    images: List[np.ndarray],
    ground_truth: List[str],
    preprocessing_steps: List[str] = None
) -> Dict[str, Any]:
    """모델을 평가합니다."""
    start_time = time.time()
    predictions = []
    
    for img in images:
        # 전처리 적용
        processed_img = img
        if preprocessing_steps:
            for step in preprocessing_steps:
                if step == 'sharpening':
                    preprocessor = SharpeningPreprocessor()
                    processed_img = preprocessor(processed_img)
                # 다른 전처리 단계들 추가 가능
        
        # 예측 수행
        pred = model(processed_img)
        predictions.extend(pred)
    
    inference_time = time.time() - start_time
    
    # 정확도 계산
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    # 문자 수준 정확도 계산
    total_chars = sum(len(g) for g in ground_truth)
    correct_chars = sum(sum(1 for c1, c2 in zip(p, g) if c1 == c2) 
                       for p, g in zip(predictions, ground_truth))
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    return {
        'metrics': {
            'accuracy': accuracy,
            'char_accuracy': char_accuracy,
            'inference_time': inference_time
        },
        'predictions': predictions
    }

def main():
    # 설정 로드
    config = load_config()
    
    # 테스트 데이터 로드
    test_images = load_test_images(config['data']['test_dir'])
    # TODO: ground_truth 데이터 로드 구현 필요
    
    # 모델 초기화
    models = {
        'easyocr': EasyOCRModel(),
        # 다른 모델들 추가 가능
    }
    
    # 전처리 단계 조합
    preprocessing_combinations = [
        [],  # 전처리 없음
        ['sharpening'],
        # 다른 전처리 조합 추가 가능
    ]
    
    # 모든 조합에 대해 평가 수행
    for model_name, model in models.items():
        for preprocess_steps in preprocessing_combinations:
            # 평가 설정 생성
            eval_config = create_evaluation_config(
                model_name=model_name,
                preprocessing_steps=preprocess_steps,
                use_gpu=config['hardware']['use_gpu']
            )
            
            # 평가 수행
            results = evaluate_model(
                model=model,
                images=test_images,
                ground_truth=[],  # TODO: ground_truth 데이터 필요
                preprocessing_steps=preprocess_steps
            )
            
            # 결과 저장
            save_evaluation_results(results, eval_config)
    
    # 전체 결과 분석 및 보고서 생성
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("평가 완료! 결과는 results 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main() 