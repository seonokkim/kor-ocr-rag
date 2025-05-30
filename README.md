# Korean OCR Evaluation Framework

This repository provides a comprehensive framework for evaluating the performance of Korean OCR models.

## Key Features

- Support for multiple OCR models:
  - EasyOCR
  - PaddleOCR (Optional - requires successful installation)
  - Tesseract (Planned)
  - YOLO-based OCR (Planned)

- Modular image preprocessing steps:
  - Sharpening
  - Denoising
  - Binarization

- GPU/CPU support
- Comprehensive evaluation metrics:
  - Item Accuracy (Bounding box matching)
  - Character Accuracy
  - Inference Time
  - Accuracy by Text Type
  - Accuracy by Location (Top/Middle/Bottom)
  - Accuracy by Text Length (Short/Medium/Long)
  - Accuracy by Bounding Box Size (Small/Medium/Large)
  - Text Similarity (Normalized Levenshtein, ROUGE, BLEU)

- Structured results saving (JSON files with sequential numbering)
- Performance report generation (CSV files with sequential numbering)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/seonokkim/kor-ocr-rag
   cd kor-ocr-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: PaddleOCR installation can sometimes be tricky due to dependencies. If you encounter issues, the script is designed to skip PaddleOCR evaluation if the module is not available.*

## Usage

1. Organize your data:
   Place your test images in the directory specified by `data.test_dir` and corresponding label JSON files in `data.label_dir` in `configs/default_config.yaml`. The current structure expects images in `test_dir/images/...` and labels in `label_dir/labels/...`.

2. Configure your evaluation:
   Modify `configs/default_config.yaml` to select the model(s), preprocessing steps, and hardware settings (GPU/CPU) you want to use.

3. Run the evaluation script:
   ```bash
   python src/run_evaluation.py
   ```
   This will perform the evaluation based on your configuration, save detailed results as JSON files with sequential numbering, and generate a performance report CSV file with sequential numbering in the `results/` directory.

## Project Structure

```
kor-ocr-rag/
├── data/                      # Data folder (images, labels)
├── src/
│   ├── models/               # OCR model implementations
│   ├── preprocessing/        # Image preprocessing modules
│   ├── evaluation/          # Performance evaluation module
│   └── utils/               # Utility functions
├── tests/                   # Test code
├── configs/                 # Configuration files
├── results/                 # Experiment results and reports
└── requirements.txt         # Dependency packages
```

## License

MIT License 