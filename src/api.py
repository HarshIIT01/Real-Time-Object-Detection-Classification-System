"""
Production REST API supporting both Classification and Detection.

Fix H: TF classification preprocessing is now loaded dynamically from
configs/config.yaml (via src.model.get_preprocess_fn) instead of being
hardcoded to MobileNetV2.  The preprocess function is resolved once at
startup and reused per request — no per-request config loading overhead.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import os
import time
import json
import logging

app = FastAPI(
    title="Real-Time Object Detection API",
    description="Production API for classification and detection inference",
    version="2.0.0",
)

logger = logging.getLogger("RT_ObjectDetection")

# Global model holders
TF_MODEL = None
YOLO_MODEL = None
CLASS_NAMES = []
BACKEND = "tf"
# Fix H: resolved once at startup, not hardcoded
_TF_PREPROCESS_FN = None


def _load_class_names(model_dir):
    """Load class names from JSON file."""
    for d in [model_dir, os.path.dirname(model_dir), "outputs/models"]:
        path = os.path.join(d, "class_names.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return [f"Class_{i}" for i in range(80)]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global TF_MODEL, YOLO_MODEL, CLASS_NAMES, BACKEND, _TF_PREPROCESS_FN

    # Try YOLO first with environment variable or fallback
    yolo_path = os.environ.get("YOLO_MODEL_PATH", os.environ.get("MODEL_PATH", "outputs/models/yolov8_run/weights/best.pt"))
    tf_path = os.environ.get("TF_MODEL_PATH", os.environ.get("MODEL_PATH", "outputs/models/classification/best_model.keras"))

    if os.path.exists(yolo_path) and yolo_path.endswith('.pt'):
        from ultralytics import YOLO
        YOLO_MODEL = YOLO(yolo_path)
        BACKEND = "yolo"
        CLASS_NAMES = list(YOLO_MODEL.names.values()) if hasattr(YOLO_MODEL, 'names') else []
        logger.info(f"YOLO model loaded: {yolo_path}")

    elif os.path.exists(tf_path):
        import tensorflow as tf
        TF_MODEL = tf.keras.models.load_model(tf_path, compile=False)
        BACKEND = "tf"
        CLASS_NAMES = _load_class_names(os.path.dirname(tf_path))
        logger.info(f"TF model loaded: {tf_path}")

        # Fix H: resolve preprocessing from config — not hardcoded
        try:
            from src.utils import load_config
            from src.model import get_preprocess_fn
            cfg = load_config()
            backbone = cfg['model'].get('backbone', 'MobileNetV2')
            _TF_PREPROCESS_FN = get_preprocess_fn(backbone)
            logger.info(f"TF preprocessing function resolved for backbone: {backbone}")
        except Exception as e:
            # Graceful fallback so API still starts even without config
            logger.warning(
                f"Could not resolve preprocessing from config ({e}). "
                f"Falling back to MobileNetV2 preprocessing."
            )
            import tensorflow as _tf
            _TF_PREPROCESS_FN = _tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        logger.warning("No model found. API running in demo mode.")


@app.get("/")
def root():
    return {"message": "Object Detection API v2.0", "backend": BACKEND, "classes": len(CLASS_NAMES)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": TF_MODEL is not None or YOLO_MODEL is not None}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    t = time.time()

    if BACKEND == "yolo" and YOLO_MODEL:
        results = YOLO_MODEL.predict(img, conf=conf, iou=iou, verbose=False)
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": CLASS_NAMES[int(box.cls)] if int(box.cls) < len(CLASS_NAMES) else f"Class_{int(box.cls)}",
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            })
        latency = (time.time() - t) * 1000
        return {"detections": detections, "count": len(detections), "latency_ms": round(latency, 2)}

    elif TF_MODEL:
        import tensorflow as tf
        # Dynamically determine the model's required input height and width
        input_shape = TF_MODEL.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        h = input_shape[1] if (len(input_shape) > 1 and input_shape[1] is not None) else 224
        w = input_shape[2] if (len(input_shape) > 2 and input_shape[2] is not None) else 224
        
        resized = cv2.resize(img, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(rgb, 0).astype(np.float32)

        # Fix H: use the backbone-specific preprocessing resolved at startup
        if _TF_PREPROCESS_FN is not None:
            arr = _TF_PREPROCESS_FN(arr)
        else:
            # Absolute fallback — should never reach here if startup succeeded
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

        preds = TF_MODEL.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        latency = (time.time() - t) * 1000
        return {
            "prediction": {"class": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}",
                           "confidence": float(preds[0][idx])},
            "latency_ms": round(latency, 2),
        }

    raise HTTPException(503, "No model loaded")


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
