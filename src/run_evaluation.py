import os
import time
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any
import json

from models import EasyOCRModel
from preprocessing import (
    SharpeningPreprocessor,
    DenoisingPreprocessor,
    BinarizationPreprocessor
)
from utils.evaluation_utils import (
    create_evaluation_config,
    save_evaluation_results,
    load_all_results,
    generate_performance_report
)

def load_test_data(config: Dict[str, Any]) -> tuple:
    """테스트 데이터를 로드합니다."""
    images = []
    ground_truth = []
    
    data_dir = config['data']['test_dir']
    label_dir = config['data']['label_dir']

    # 이미지와 레이블 파일 매칭
    for img_path in Path(data_dir).glob("*.jpg"):
        label_filename = img_path.name.replace(".jpg", ".json")
        json_path = Path(label_dir) / label_filename
        
        if json_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                with open(json_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                
                # JSON 데이터에서 'annotation.text' 값들을 추출하여 합치기
                text = " ".join([anno.get('annotation.text', '') for anno in label_data.get('annotations', [])])
                ground_truth.append(text)
                images.append(img)
            else:
                print(f"Warning: Could not load image {img_path}")
        else:
            print(f"Warning: No corresponding JSON file found for {img_path.name} in {label_dir}")
    
    return images, ground_truth

def get_preprocessing_pipeline(steps: List[str]) -> List[Any]:
    """전처리 파이프라인을 생성합니다."""
    pipeline = []
    for step in steps:
        if step == 'sharpening':
            pipeline.append(SharpeningPreprocessor())
        elif step == 'denoising':
            pipeline.append(DenoisingPreprocessor())
        elif step == 'binarization':
            pipeline.append(BinarizationPreprocessor())
    return pipeline

def evaluate_combination(
    model,
    images: List[np.ndarray],
    ground_truth: List[str],
    preprocessing_steps: List[str]
) -> Dict[str, Any]:
    """특정 모델과 전처리 조합을 평가합니다."""
    start_time = time.time()
    predictions = []
    
    # 전처리 파이프라인 생성
    pipeline = get_preprocessing_pipeline(preprocessing_steps)
    
    for img in images:
        # 전처리 적용
        processed_img = img
        for preprocessor in pipeline:
            processed_img = preprocessor(processed_img)
        
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
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 테스트 데이터 로드
    test_images, ground_truth = load_test_data(config)
    print(f"로드된 테스트 이미지 수: {len(test_images)}")
    
    # 모델 초기화
    models = {
        'easyocr': EasyOCRModel(),
        # 다른 모델들 추가 가능
    }
    
    # 전처리 조합 생성
    preprocessing_combinations = [
        [],  # 전처리 없음
        ['sharpening'],
        ['denoising'],
        ['binarization'],
        ['sharpening', 'denoising'],
        ['sharpening', 'binarization'],
        ['denoising', 'binarization'],
        ['sharpening', 'denoising', 'binarization']
    ]
    
    # 모든 조합에 대해 평가 수행
    for model_name, model in models.items():
        print(f"\n모델 평가 중: {model_name}")
        for preprocess_steps in preprocessing_combinations:
            print(f"전처리 단계: {preprocess_steps if preprocess_steps else '없음'}")
            
            # 평가 설정 생성
            eval_config = create_evaluation_config(
                model_name=model_name,
                preprocessing_steps=preprocess_steps,
                use_gpu=config['hardware']['use_gpu']
            )
            
            # 평가 수행
            results = evaluate_combination(
                model=model,
                images=test_images,
                ground_truth=ground_truth,
                preprocessing_steps=preprocess_steps
            )
            
            # 결과 저장
            save_evaluation_results(results, eval_config)
            
            # 중간 결과 출력
            print(f"정확도: {results['metrics']['accuracy']:.4f}")
            print(f"문자 정확도: {results['metrics']['char_accuracy']:.4f}")
            print(f"추론 시간: {results['metrics']['inference_time']:.2f}초")
    
    # 전체 결과 분석 및 보고서 생성
    print("\n전체 결과 분석 중...")
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("평가 완료! 결과는 results 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main() 