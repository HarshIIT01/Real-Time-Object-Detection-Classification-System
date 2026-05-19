# Next-Generation Real-Time Object Detection & Classification System

## Executive Summary
A production-ready, high-accuracy Real-Time Object Detection and Classification System with dual pipelines: **YOLOv8** for detection and **MobileNetV2/EfficientNet** for classification. Designed for maximum accuracy, low latency, and edge deployment readiness.

## Architecture

The system supports two major pipelines:

### 1. Classification Pipeline (TF/Keras)
- **Backbones**: MobileNetV2, EfficientNetB0, EfficientNetB3, EfficientNetV2S
- **Training**: Two-phase transfer learning (head → progressive unfreezing)
- **LR Schedule**: Warmup + Cosine Annealing
- **Augmentation**: MixUp, CutMix, Random Erasing, geometric + photometric
- **Head**: Configurable multi-layer dense head with BatchNorm + Dropout

### 2. Object Detection Pipeline (YOLOv8)
- **Models**: YOLOv8n (fast) / YOLOv8s (balanced) / YOLOv8m (accurate) / YOLOv8l / YOLOv8x
- **Training**: 640×640 input, cosine LR, mosaic + MixUp + copy-paste augmentation
- **Datasets**: COCO, Open Images V7, Roboflow Universe, or local custom
- **Inference**: Configurable confidence/IoU thresholds, Test-Time Augmentation (TTA)

### Key Accuracy Improvements
- ✅ Proper 640px training for detection (was 224px)
- ✅ Cosine LR with warmup (was static LR)
- ✅ YOLOv8s default (was nano — 30% more accurate)
- ✅ Mosaic + MixUp + Copy-Paste augmentation
- ✅ 100 epochs with patience-30 early stopping
- ✅ BatchNorm frozen during fine-tuning
- ✅ Label smoothing + weight decay
- ✅ Configurable all 25+ YOLO hyperparameters

## Directory Structure
```
real_time_object_detection/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── src/
│   ├── __init__.py           # Package initializer
│   ├── train.py              # Training pipeline (classification + detection)
│   ├── predict.py            # Real-time inference with webcam
│   ├── model.py              # Classification model builder
│   ├── data_loader.py        # tf.data pipeline with augmentation
│   ├── dataset_manager.py    # COCO/OpenImages/Roboflow integration
│   ├── augmentations.py      # MixUp, CutMix, Random Erasing
│   ├── evaluate.py           # mAP, confusion matrix, Grad-CAM
│   ├── benchmark.py          # Latency/FPS benchmarking
│   ├── export.py             # TFLite/ONNX/TorchScript export
│   ├── api.py                # FastAPI REST endpoint
│   └── gui_app.py            # Streamlit dashboard
├── tests/
│   └── test_model.py         # Comprehensive test suite
├── configs/                  # YAML configurations
├── outputs/                  # Trained models & evaluation results
├── logs/                     # TensorBoard + system logs
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

### Fix OpenCV GUI issues
If `cv2.imshow` fails (common when `roboflow` installs headless OpenCV):
```bash
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python
```

## Usage

### 1. Configure (`configs/config.yaml`)

Choose your task and model:
```yaml
system:
  task: "detection"            # or "classification"

model:
  yolo_model: "yolov8s.pt"    # n/s/m/l/x
  backbone: "MobileNetV2"     # For classification

data:
  detection_dataset: "coco8.yaml"   # coco.yaml, open-images-v7.yaml, roboflow
  detection_imgsz: 640              # Must be multiple of 32
```

### 2. Train
```bash
python -m src.train --config configs/config.yaml
```

### 3. Real-Time Inference
```bash
# Detection (YOLOv8)
python -m src.predict --model outputs/models/yolov8_run/weights/best.pt --conf 0.25 --iou 0.45

# Detection with Test-Time Augmentation (slower, more accurate)
python -m src.predict --model outputs/models/yolov8_run/weights/best.pt --tta

# Classification
python -m src.predict --model outputs/models/best_model.keras
```

### 4. Evaluate
```bash
# Detection — reports mAP@50, mAP@50:95
python -m src.evaluate --model outputs/models/yolov8_run/weights/best.pt --task detection

# Classification — confusion matrix, per-class report
python -m src.evaluate --model outputs/models/best_model.keras --task classification
```

### 5. Benchmark
```bash
python -m src.benchmark --model outputs/models/yolov8_run/weights/best.pt --runs 200
```

### 6. Export
```bash
# YOLO → ONNX
python -m src.export --model outputs/models/yolov8_run/weights/best.pt --yolo-format onnx

# TF → TFLite (FP16)
python -m src.export --model outputs/models/best_model.keras --quantize fp16
```

### 7. API Server
```bash
# Set model path via environment variable
set MODEL_PATH=outputs/models/yolov8_run/weights/best.pt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 8. GUI Dashboard
```bash
streamlit run src/gui_app.py
```

### 9. Run Tests
```bash
python -m pytest tests/ -v
```

## Scaling Up

| Goal | Config Change |
|------|--------------|
| Quick test | `detection_dataset: "coco8.yaml"` |
| Full COCO | `detection_dataset: "coco.yaml"` |
| Open Images V7 | `detection_dataset: "open-images-v7.yaml"` |
| Roboflow custom | `detection_dataset: "roboflow"` + API key |
| Higher accuracy | `yolo_model: "yolov8m.pt"` or `"yolov8l.pt"` |
| Faster inference | `yolo_model: "yolov8n.pt"` |
| Edge deployment | Export to TFLite INT8 or TensorRT |
