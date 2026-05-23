"""
Real-time inference engine supporting both Classification and Detection.

Features:
  - Auto-detects model type from file extension (.pt = YOLO, .keras/.h5 = TF)
  - Async threaded video capture for maximum FPS
  - Test-Time Augmentation (TTA) support for YOLO
  - Loads class names from training artifacts
  - Professional OSD (On-Screen Display) with metrics
"""
import cv2
import time
import os
import json
import argparse
import logging

import numpy as np
import tensorflow as tf
from threading import Thread

logger = logging.getLogger("RT_ObjectDetection")


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO CAPTURE
# ═══════════════════════════════════════════════════════════════════════════

class VideoStream:
    """
    Asynchronous multi-threaded video stream.
    Decouples frame capture from inference to maximize throughput.
    """
    def __init__(self, src=0, resolution=None):
        self.stream = cv2.VideoCapture(src)
        if resolution:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                return
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

    @property
    def fps(self):
        return self.stream.get(cv2.CAP_PROP_FPS) or 30


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_opencv_gui():
    """Fail fast if OpenCV lacks GUI support (headless build)."""
    try:
        cv2.namedWindow("__gui_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_test__")
    except Exception as e:
        raise RuntimeError(
            "OpenCV GUI (cv2.imshow) unavailable.\n"
            "Fix on Windows:\n"
            "  pip uninstall -y opencv-python-headless\n"
            "  pip install --force-reinstall opencv-python\n"
            f"Error: {e}"
        ) from e


def load_class_names(model_path):
    """
    Loads class names from a JSON file saved alongside the model.
    Falls back to generic names if not found.
    """
    search_dirs = [
        os.path.dirname(model_path),
        os.path.join(os.path.dirname(model_path), ".."),
        "outputs/models",
    ]

    for d in search_dirs:
        path = os.path.join(d, "class_names.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                names = json.load(f)
            logger.info(f"Loaded {len(names)} class names from {path}")
            return names

    logger.warning("No class_names.json found — using generic labels.")
    return None


def detect_backend(model_path, explicit_backend='auto'):
    """Auto-detect inference backend from model file extension."""
    if explicit_backend != 'auto':
        return explicit_backend

    ext = os.path.splitext(model_path)[1].lower()
    mapping = {
        '.pt': 'yolo',
        '.pth': 'yolo',
        '.engine': 'yolo',
        '.onnx': 'yolo',      # Ultralytics can load ONNX
        '.h5': 'tf',
        '.keras': 'tf',
        '.tflite': 'tflite',
    }
    backend = mapping.get(ext, 'tf')
    logger.info(f"Auto-detected backend: '{backend}' for extension '{ext}'")
    return backend


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_inference_model(model_path, backend='tf'):
    """Loads the inference model based on backend type."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if backend == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        logger.info(f"TFLite model loaded: {model_path}")
        return interpreter
    elif backend == 'tf':
        # compile=False avoids deserializing optimizer/schedule objects,
        # which can break when older training artifacts used non-registered
        # custom learning rate schedules.
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"Keras model loaded (compile=False): {model_path}")
        return model

    else:
        raise ValueError(f"Unsupported backend: {backend}")


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def predict_tf(model, img_array):
    """Run TensorFlow/Keras inference."""
    return model.predict(img_array, verbose=0)


def predict_tflite(interpreter, img_array):
    """Run TFLite inference with proper quantization handling."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Handle quantized input
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        img_array = (img_array / scale + zero_point).astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if needed
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output


# ═══════════════════════════════════════════════════════════════════════════
#  ON-SCREEN DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def draw_osd(frame, fps, latency_ms, label=None, confidence=None, det_count=None):
    """Draw professional metrics overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Metrics
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if label and confidence is not None:
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"{label}: {confidence:.1%}", (w // 3, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if det_count is not None:
        cv2.putText(frame, f"Objects: {det_count}", (w // 3, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Quit instruction
    cv2.putText(frame, "Press 'Q' to quit", (w - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN REAL-TIME LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_realtime(model_path, backend='auto', source=0, class_names=None,
                 conf_threshold=0.25, iou_threshold=0.45, use_tta=False):
    """
    Main real-time inference loop.

    Args:
        model_path: Path to model file
        backend: 'auto', 'tf', 'tflite', or 'yolo'
        source: Webcam index or video path
        class_names: List of class names (auto-loaded if None)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        use_tta: Enable Test-Time Augmentation (YOLO only)
    """
    _ensure_opencv_gui()

    backend = detect_backend(model_path, backend)

    # Load class names if not provided
    if class_names is None:
        class_names = load_class_names(model_path)

    # Start video stream
    vs = VideoStream(src=source).start()
    time.sleep(1.5)  # Camera warmup

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    window_name = "Real-Time Inference"

    # ── YOLO Detection ──
    if backend == 'yolo':
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"YOLOv8 loaded: {model_path}")
        logger.info(f"Conf: {conf_threshold}, IoU: {iou_threshold}, TTA: {use_tta}")
        window_name = "YOLOv8 Real-Time Detection"

        while True:
            frame = vs.read()
            if frame is None:
                break

            t1 = time.time()
            results = model.predict(
                source=frame,
                show=False,
                verbose=False,
                conf=conf_threshold,
                iou=iou_threshold,
                augment=use_tta,
            )
            latency = (time.time() - t1) * 1000

            annotated = results[0].plot()
            det_count = len(results[0].boxes) if results[0].boxes is not None else 0

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            annotated = draw_osd(annotated, fps, latency, det_count=det_count)
            cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── TF/TFLite Classification ──
    else:
        model = load_inference_model(model_path, backend)
        window_name = "Real-Time Classification"

        # Determine preprocessing
        from src.model import get_preprocess_fn
        try:
            from src.utils import load_config
            cfg = load_config()
            backbone = cfg['model']['backbone']
        except Exception:
            backbone = "MobileNetV2"

        preprocess_fn = get_preprocess_fn(backbone)

        while True:
            frame = vs.read()
            if frame is None:
                break

            t1 = time.time()

            # Preprocess
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(rgb, axis=0).astype(np.float32)
            img_array = preprocess_fn(img_array)

            # Predict
            if backend == 'tflite':
                preds = predict_tflite(model, img_array)
            else:
                preds = predict_tf(model, img_array)

            latency = (time.time() - t1) * 1000

            class_idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][class_idx])
            label = class_names[class_idx] if class_names and class_idx < len(class_names) else f"Class_{class_idx}"

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            display = draw_osd(frame, fps, latency, label=label, confidence=confidence)
            cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    vs.stop()
    cv2.destroyAllWindows()
    logger.info("Inference stopped.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.h5/.keras/.tflite/.pt)")
    parser.add_argument("--backend", type=str, default="auto", choices=['auto', 'tf', 'tflite', 'yolo'])
    parser.add_argument("--source", type=int, default=0, help="Webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()

    run_realtime(
        model_path=args.model,
        backend=args.backend,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_tta=args.tta,
    )
