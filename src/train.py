import os
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any, List
import time
import json

# 다른 모델의 학습 기능을 import
# from models.tesseract import TesseractModel # 예시
# from models.paddleocr import PaddleOCRModel # 예시

def load_training_data(train_dir: str) -> tuple:
    """학습 데이터를 로드합니다 (하위 폴더 포함)."""
    images = []
    labels = []
    
    # 학습 데이터 로딩 로직 (하위 폴더 탐색)
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                # 예: 이미지 파일과 같은 이름의 .json 파일 로드 (상대 경로 유지)
                label_path = (Path(train_dir) / Path(root).relative_to(train_dir) / file).with_suffix('.json').parent.parent / Path('labels') / Path(root).relative_to(Path(train_dir) / 'images') / file.replace('.jpg', '.json')
                
                # 레이블 파일 경로 디버깅 출력
                # print(f"Checking label path: {label_path}")

                if label_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                         with open(label_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            # JSON 데이터에서 'annotation.text' 값들을 추출하여 합치기
                            text = " ".join([anno.get('annotation.text', '') for anno in label_data.get('annotations', [])])
                            labels.append(text)
                            images.append(img)
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: No corresponding JSON file found for {img_path} at {label_path}")

    return images, labels

def train_model(model_name: str, train_data: tuple, config: Dict[str, Any], save_path: str):
    """주어진 데이터로 모델을 학습하고 저장합니다."""
    print(f"\n모델 학습 시작: {model_name}")
    
    images, labels = train_data
    
    if not images:
        print("학습 데이터가 없습니다. 학습을 건너뜁니다.")
        return

    # 실제 학습 로직 구현
    # EasyOCR은 사용자 학습을 직접 지원하는 API가 제한적입니다. 
    # 여기서는 개념적인 학습 구조를 제시하며, 실제 학습은 해당 라이브러리의 가이드를 참고하거나
    # 사용자 학습을 지원하는 다른 모델(예: Tesseract, PaddleOCR)을 사용해야 합니다.

    if model_name == 'tesseract':
        print("Tesseract 모델 학습 기능은 별도 구현이 필요합니다 (tesseract train tool 활용).")
        # Tesseract 학습 도구를 사용하여 모델 학습 및 save_path에 저장하는 로직 구현
        pass
    elif model_name == 'paddleocr':
        print("PaddleOCR 모델 학습 기능은 별도 구현이 필요합니다 (PaddleOCR 트레이닝 프레임워크 활용).")
        # PaddleOCR 학습 도구를 사용하여 모델 학습 및 save_path에 저장하는 로직 구현
        pass
    # 다른 모델 학습 로직 추가
    else:
        print(f"{model_name} 모델은 사용자 학습 기능을 지원하지 않거나 학습 코드가 구현되지 않았습니다.")
        print(f"학습된 모델을 {save_path}에 저장하지 않습니다.")
        return
    
    print(f"{model_name} 모델 학습 완료. 학습된 모델을 {save_path}에 저장했습니다.")

def main():
    # 설정 로드
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 학습 데이터 로드
    train_images, train_labels = load_training_data(config['data']['train_dir'])
    print(f"로드된 학습 이미지 수: {len(train_images)}")

    # 학습된 모델 저장 경로
    trained_model_dir = "trained_models"
    os.makedirs(trained_model_dir, exist_ok=True)
    
    # 설정 파일에 정의된 학습 가능한 모델들에 대해 학습 수행
    learnable_models = [m for m in config['models']['available'] if m in ['tesseract', 'paddleocr']] # 사용자 학습 지원 모델 목록

    for model_name in learnable_models:
        model_save_path = os.path.join(trained_model_dir, f'{model_name}_korean')
        train_model(model_name, (train_images, train_labels), config, model_save_path)

if __name__ == "__main__":
    main() 