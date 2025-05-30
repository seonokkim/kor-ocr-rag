import os
import time
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from Levenshtein import distance as levenshtein_distance
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Attempt to import PaddleOCR module
try:
    from models import EasyOCRModel, PaddleOCRModel
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"\nWarning: Could not import PaddleOCR module - {str(e)}")
    print("PaddleOCR model will be excluded from evaluation.")
    from models import EasyOCRModel
    PADDLEOCR_AVAILABLE = False

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

def bbox_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def convert_bbox_to_x1y1x2y2(bbox, fmt='easyocr'):
    """Convert bounding box format to [x1, y1, x2, y2]."""
    if fmt == 'easyocr':
        # EasyOCR format is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] or similar quadrilateral
        # We need the min/max x and y
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif fmt == 'json':
        # JSON format is [x, y, width, height]
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    else:
        raise ValueError(f"Unknown bounding box format: {fmt}")

def load_test_data(config: Dict[str, Any]) -> tuple:
    """Loads test data (including subfolders, loads all annotations)."""
    images = []
    ground_truth_annotations = [] # Store list of all annotations instead of list of texts
    
    data_dir = config['data']['test_dir']
    label_dir = config['data']['label_dir']

    # Match images and label files (explore subfolders)
    for root, _, files in os.walk(Path(data_dir) / 'images'): # Explore from images/ subfolder
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                
                # Relative path of the image file based on images/
                relative_img_sub_path = img_path.relative_to(Path(data_dir) / 'images')
                
                # Label file path (based on label_dir)
                json_path = Path(label_dir) / 'labels' / relative_img_sub_path.parent / relative_img_sub_path.name.replace('.jpg', '.json')

                # Debug print for label file path
                # print(f"Checking test label path: {json_path}")

                if json_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                        
                        # Store list of all annotations
                        ground_truth_annotations.append(label_data.get('annotations', []))
                        images.append(img)
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: No corresponding JSON file found for {img_path.name} at {json_path}")
    
    return images, ground_truth_annotations

def get_preprocessing_pipeline(steps: List[str]) -> List[Any]:
    """Creates a preprocessing pipeline."""
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
    ground_truth: List[List[Dict[str, Any]]],
    preprocessing_steps: List[str]
) -> Dict[str, Any]:
    """Evaluates a specific model and preprocessing combination."""
    start_time = time.time()
    all_predictions = []
    
    # Create preprocessing pipeline
    pipeline = get_preprocessing_pipeline(preprocessing_steps)
    
    for img in images:
        # Apply preprocessing
        processed_img = img
        for preprocessor in pipeline:
            processed_img = preprocessor(processed_img)
        
        # Perform prediction
        pred = model(processed_img)
        all_predictions.append(pred)
    
    inference_time = time.time() - start_time
    
    # Calculate accuracy (bounding box based matching)
    total_items = 0
    matched_items = 0
    total_chars = 0
    matched_chars = 0
    
    # Counters for additional metrics
    type_metrics = {}  # Accuracy by text type
    region_metrics = {}  # Accuracy by location (top/middle/bottom)
    length_metrics = {}  # Accuracy by text length (short/medium/long)
    size_metrics = {}  # Accuracy by bounding box size (small/medium/large)
    
    # Variables for full text comparison
    total_levenshtein_distance = 0
    total_gt_length = 0
    rouge = Rouge()
    total_rouge_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    total_rouge_count = 0
    
    # Variables for BLEU score
    total_bleu_score = 0
    total_bleu_count = 0
    smoothing = SmoothingFunction().method1
    
    for pred_list, gt_annotations in zip(all_predictions, ground_truth):
        # Extract text and bounding boxes from Ground Truth annotations
        gt_texts = [anno.get('annotation.text', '') for anno in gt_annotations]
        gt_boxes = [convert_bbox_to_x1y1x2y2(anno.get('annotation.bbox', []), fmt='json') 
                   for anno in gt_annotations]
        gt_types = [anno.get('annotation.ttype', 'unknown') for anno in gt_annotations]
        
        # Generate strings for full text comparison
        gt_full_text = ' '.join(gt_texts)
        pred_full_text = ' '.join([text for text, _ in pred_list])
        
        # Calculate Levenshtein distance
        total_levenshtein_distance += levenshtein_distance(gt_full_text, pred_full_text)
        total_gt_length += len(gt_full_text)
        
        # Calculate ROUGE score
        try:
            if gt_full_text and pred_full_text:
                rouge_scores = rouge.get_scores(pred_full_text, gt_full_text)[0]
                for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                    total_rouge_scores[metric] += rouge_scores[metric]['f']
                total_rouge_count += 1
        except Exception as e:
            print(f"Warning: Error calculating ROUGE score - {str(e)}")
        
        # Calculate BLEU score
        try:
            if gt_full_text and pred_full_text:
                # Split sentences into words
                reference = [gt_full_text.split()]
                candidate = pred_full_text.split()
                
                # Calculate BLEU score (1-gram, 2-gram, 3-gram, 4-gram)
                weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
                bleu_scores = []
                
                for weight in weights:
                    score = sentence_bleu(reference, candidate, weights=weight, smoothing_function=smoothing)
                    bleu_scores.append(score)
                
                # Calculate average BLEU score
                total_bleu_score += sum(bleu_scores) / len(bleu_scores)
                total_bleu_count += 1
        except Exception as e:
            print(f"Warning: Error calculating BLEU score - {str(e)}")
        
        total_items += len(gt_texts)
        
        # Match predictions with Ground Truth
        matched_gt_indices = set()
        for pred_text, pred_box in pred_list:
            best_iou = 0
            best_gt_idx = -1
            
            # Find the Ground Truth with the highest IoU
            for i, (gt_text, gt_box) in enumerate(zip(gt_texts, gt_boxes)):
                if i in matched_gt_indices:
                    continue
                    
                iou = bbox_iou(pred_box, gt_box)
                if iou > best_iou and iou > 0.5:  # IoU threshold
                    best_iou = iou
                    best_gt_idx = i
            
            # If matched
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                matched_items += 1
                
                # Calculate character-level accuracy
                gt_text = gt_texts[best_gt_idx]
                total_chars += len(gt_text)
                matched_chars += sum(1 for c1, c2 in zip(pred_text, gt_text) if c1 == c2)
                
                # Accuracy by text type
                gt_type = gt_types[best_gt_idx]
                if gt_type not in type_metrics:
                    type_metrics[gt_type] = {'total': 0, 'matched': 0}
                type_metrics[gt_type]['total'] += 1
                if pred_text == gt_text:
                    type_metrics[gt_type]['matched'] += 1
                
                # Accuracy by location (top/middle/bottom)
                y_center = (gt_box[1] + gt_box[3]) / 2
                region = 'top' if y_center < 0.33 else 'middle' if y_center < 0.66 else 'bottom'
                if region not in region_metrics:
                    region_metrics[region] = {'total': 0, 'matched': 0}
                region_metrics[region]['total'] += 1
                if pred_text == gt_text:
                    region_metrics[region]['matched'] += 1
                
                # Accuracy by text length
                length = len(gt_text)
                length_key = 'short' if length <= 2 else 'medium' if length <= 5 else 'long'
                if length_key not in length_metrics:
                    length_metrics[length_key] = {'total': 0, 'matched': 0}
                length_metrics[length_key]['total'] += 1
                if pred_text == gt_text:
                    length_metrics[length_key]['matched'] += 1
                
                # Accuracy by bounding box size
                box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                size_key = 'small' if box_area < 1000 else 'medium' if box_area < 5000 else 'large'
                if size_key not in size_metrics:
                    size_metrics[size_key] = {'total': 0, 'matched': 0}
                size_metrics[size_key]['total'] += 1
                if pred_text == gt_text:
                    size_metrics[size_key]['matched'] += 1
    
    # Calculate final accuracies
    item_accuracy = float(matched_items) / total_items if total_items > 0 else 0.0
    char_accuracy = float(matched_chars) / total_chars if total_chars > 0 else 0.0
    
    # Calculate additional metrics
    type_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                      for k, v in type_metrics.items()}
    region_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                        for k, v in region_metrics.items()}
    length_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                        for k, v in length_metrics.items()}
    size_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                      for k, v in size_metrics.items()}
    
    # Calculate full text comparison metrics
    normalized_levenshtein = 1.0 - (float(total_levenshtein_distance) / total_gt_length) if total_gt_length > 0 else 0.0
    rouge_scores = {k: float(v) / total_rouge_count if total_rouge_count > 0 else 0.0 
                   for k, v in total_rouge_scores.items()}
    bleu_score = float(total_bleu_score) / total_bleu_count if total_bleu_count > 0 else 0.0

    return {
        'metrics': {
            'item_accuracy': item_accuracy,
            'char_accuracy': char_accuracy,
            'inference_time': inference_time,
            'type_accuracies': type_accuracies,
            'region_accuracies': region_accuracies,
            'length_accuracies': length_accuracies,
            'size_accuracies': size_accuracies,
            'text_similarity': {
                'normalized_levenshtein': normalized_levenshtein,
                'rouge_scores': rouge_scores,
                'bleu_score': bleu_score
            }
        },
        'predictions': all_predictions
    }

def main():
    # Load configuration
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    test_images, ground_truth = load_test_data(config)
    print(f"Loaded {len(test_images)} test images.")
    
    # Trained models directory
    trained_model_dir = "trained_models"

    # Initialize models and set up evaluation targets
    evaluation_targets = {}

    # 1. Add base models
    base_model_name = config['models']['selected']
    # Initialize EasyOCR model (use_gpu argument is used)
    evaluation_targets['base_easyocr'] = EasyOCRModel(use_gpu=config['hardware']['use_gpu'])
    
    # Initialize PaddleOCR model (skip if module is not available or initialization fails)
    if PADDLEOCR_AVAILABLE:
        try:
            evaluation_targets['base_paddleocr'] = PaddleOCRModel(use_gpu=config['hardware']['use_gpu'])
        except Exception as e:
            print(f"\nWarning: PaddleOCR model initialization failed - {str(e)}")
            print("PaddleOCR model evaluation will be skipped.")
    
    # Add other base models (if needed)

    # 2. Add trained models (if they exist)
    # Check for learnable models defined in config and attempt to load trained models
    learnable_models = ['tesseract', 'paddleocr'] # List of models supporting user training
    for model_name in learnable_models:
        trained_model_path = os.path.join(trained_model_dir, f'{model_name}_korean')
        # Check if trained model file (e.g., pytorch model file) exists
        # Needs to be adjusted according to the actual model file extension and structure
        if os.path.exists(trained_model_path): # Check if trained model directory or file exists
             try:
                 # TODO: Implement trained model loading logic
                 # Example: if model_name == 'tesseract': loaded_model = TesseractModel(model_path=trained_model_path)
                 # Example: elif model_name == 'paddleocr': loaded_model = PaddleOCRModel(model_path=trained_model_path)
                 print(f"\nWarning: Loading trained {model_name} model is not yet implemented.")
                 # evaluation_targets[f'trained_{model_name}'] = loaded_model
             except Exception as e:
                 print(f"Warning: Failed to load trained {model_name} model from {trained_model_path}: {e}")
        else:
            print(f"Info: Trained {model_name} model not found at {trained_model_path}. Skipping evaluation for this model.")

    if not evaluation_targets:
        print("No models available for evaluation. Exiting script.")
        return
    
    # Generate preprocessing combinations
    preprocessing_combinations = [
        [],  # No preprocessing
        ['sharpening'],
        ['denoising'],
        ['binarization'],
        ['sharpening', 'denoising'],
        ['sharpening', 'binarization'],
        ['denoising', 'binarization'],
        ['sharpening', 'denoising', 'binarization']
    ]
    
    # Perform evaluation for all model and preprocessing combinations
    for target_name, model in evaluation_targets.items():
        print(f"\nEvaluating model: {target_name}")
        for preprocess_steps in preprocessing_combinations:
            print(f"Preprocessing steps: {preprocess_steps if preprocess_steps else 'None'}")
            
            # Create evaluation config
            eval_config = create_evaluation_config(
                model_name=target_name, # Include base/trained info in model name
                preprocessing_steps=preprocess_steps,
                use_gpu=config['hardware']['use_gpu']
            )
            
            # Perform evaluation
            results = evaluate_combination(
                model=model,
                images=test_images,
                ground_truth=ground_truth,
                preprocessing_steps=preprocess_steps
            )
            
            # Save results
            save_evaluation_results(results, eval_config)
            
            # Print intermediate results
            print(f"Item Accuracy: {results['metrics']['item_accuracy']:.4f}")
            print(f"Character Accuracy: {results['metrics']['char_accuracy']:.4f}")
            print(f"Inference Time: {results['metrics']['inference_time']:.2f} seconds")
    
    # Analyze overall results and generate report
    print("\nAnalyzing overall results...")
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("Evaluation complete! Results can be found in the results directory.")

if __name__ == "__main__":
    main() 