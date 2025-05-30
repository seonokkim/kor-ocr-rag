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
    """테스트 데이터를 로드합니다 (하위 폴더 포함)."""
    images = []
    ground_truth = []
    
    data_dir = config['data']['test_dir']
    label_dir = config['data']['label_dir']

    # 이미지와 레이블 파일 매칭 (하위 폴더 탐색)
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                # 예: 이미지 파일과 같은 이름의 .json 파일 로드 (상대 경로 유지)
                # 테스트 데이터의 레이블 경로도 train 데이터와 동일한 구조라고 가정
                label_path = (Path(label_dir) / Path(root).relative_to(data_dir) / file).with_suffix('.json').parent.parent / Path('labels') / Path(root).relative_to(Path(data_dir) / 'images') / file.replace('.jpg', '.json')
                
                # 레이블 파일 경로 디버깅 출력
                # print(f"Checking test label path: {label_path}")

                if label_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                        
                        # JSON 데이터에서 'annotation.text' 값들을 추출하여 리스트로 반환
                        ground_truth_list = [anno.get('annotation.text', '') for anno in label_data.get('annotations', [])]
                        ground_truth.append(ground_truth_list)
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
    
    # 정확도 계산 (항목별 일치율)
    item_correct = 0
    total_items = 0
    for pred_list, gt_list in zip(predictions, ground_truth):
        total_items += len(gt_list)
        # 간단한 비교: 예측 리스트의 각 항목이 Ground Truth 리스트에 포함되어 있는지 확인
        # 실제 평가 시에는 더 정교한 매칭 로직이 필요할 수 있습니다.
        matched_gt_indices = set()
        for pred_item in pred_list:
            for i, gt_item in enumerate(gt_list):
                if i not in matched_gt_indices and pred_item == gt_item:
                    item_correct += 1
                    matched_gt_indices.add(i)
                    break # 예측 항목 하나에 대해 하나의 Ground Truth 항목만 매칭

    item_accuracy = item_correct / total_items if total_items > 0 else 0
    
    # 문자 수준 정확도 계산 (기존 로직 유지)
    total_chars = 0
    correct_chars = 0
    for pred_list, gt_list in zip(predictions, ground_truth):
        # 각 이미지의 텍스트 리스트를 단일 문자열로 합쳐서 문자 정확도 계산
        pred_text = " ".join(pred_list)
        gt_text = " ".join(gt_list)
        total_chars += len(gt_text)
        correct_chars += sum(1 for c1, c2 in zip(pred_text, gt_text) if c1 == c2)

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

    return {
        'metrics': {
            'item_accuracy': item_accuracy, # 항목별 정확도 추가
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
    
    # 학습된 모델 디렉토리
    trained_model_dir = "trained_models"

    # 모델 초기화 및 평가 대상 설정
    evaluation_targets = {}

    # 1. 기본 모델 추가
    base_model_name = config['models']['selected']
    if base_model_name == 'easyocr':
        evaluation_targets['base_easyocr'] = EasyOCRModel()
    # 다른 기본 모델 추가 (필요시)

    # 2. 학습된 모델 추가 (존재하는 경우)
    # 설정 파일에 정의된 학습 가능한 모델들을 확인하고 학습된 모델 로드 시도
    learnable_models = ['tesseract', 'paddleocr'] # 사용자 학습 지원 모델 목록
    for model_name in learnable_models:
        trained_model_path = os.path.join(trained_model_dir, f'{model_name}_korean')
        # 학습된 모델 파일 (예: pytorch 모델 파일 등) 존재 여부 확인
        # 실제 모델 파일 확장자 및 구조에 맞게 수정 필요
        if os.path.exists(trained_model_path): # 학습된 모델 디렉토리 또는 파일 존재 확인
             try:
                 # TODO: 학습된 모델 로드 로직 구현
                 # 예: if model_name == 'tesseract': loaded_model = TesseractModel(model_path=trained_model_path)
                 # 예: elif model_name == 'paddleocr': loaded_model = PaddleOCRModel(model_path=trained_model_path)
                 print(f"\nWarning: Loading trained {model_name} model is not yet implemented.")
                 # evaluation_targets[f'trained_{model_name}'] = loaded_model
             except Exception as e:
                 print(f"Warning: Failed to load trained {model_name} model from {trained_model_path}: {e}")
        else:
            print(f"Info: Trained {model_name} model not found at {trained_model_path}. Skipping evaluation for this model.")

    if not evaluation_targets:
        print("평가할 모델이 없습니다. 스크립트를 종료합니다.")
        return
    
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
    
    # 모든 모델 및 전처리 조합에 대해 평가 수행
    for target_name, model in evaluation_targets.items():
        print(f"\n모델 평가 중: {target_name}")
        for preprocess_steps in preprocessing_combinations:
            print(f"전처리 단계: {preprocess_steps if preprocess_steps else '없음'}")
            
            # 평가 설정 생성
            eval_config = create_evaluation_config(
                model_name=target_name, # 모델 이름에 기본/학습 정보 포함
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
            print(f"항목별 정확도: {results['metrics']['item_accuracy']:.4f}")
            print(f"문자 정확도: {results['metrics']['char_accuracy']:.4f}")
            print(f"추론 시간: {results['metrics']['inference_time']:.2f}초")
    
    # 전체 결과 분석 및 보고서 생성
    print("\n전체 결과 분석 중...")
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("평가 완료! 결과는 results 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main() 