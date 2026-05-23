"""
Model evaluation for Classification and Detection.

This module provides tools for evaluating both standard classification models
and YOLOv8 detection models. It includes utilities to generate comprehensive
evaluation reports, confusion matrices, and tracking metrics with full 
provenance by logging paths, configurations, and timestamps.
"""
import os
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

logger = logging.getLogger("RT_ObjectDetection")


def evaluate_classification(model_path, val_ds, class_names, output_dir="outputs/evaluation"):
    """Comprehensive classification evaluation with reports and plots."""
    os.makedirs(output_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    with open(os.path.join(output_dir, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontweight='bold')
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    logger.info(f"Accuracy: {np.mean(y_true == y_pred):.4f}")
    return report


def evaluate_detection(
    model_path,
    data_yaml=None,
    output_dir="outputs/evaluation",
    imgsz=640,
    device="cpu",
):
    """
    Evaluates a YOLOv8 detection model and reports mAP@50, mAP@50:95.

    The metrics file outputs/evaluation/detection_metrics.json documents
    the parameters and directories used so it can be reconciled with training.

    Args:
        model_path: Path to the .pt model to evaluate.
        data_yaml:  Dataset YAML (e.g. "coco128.yaml"). If None, uses the
                    dataset the model was trained on.
        output_dir: Root directory for evaluation outputs.
        imgsz:      Inference image size — should match training imgsz (640).
        device:     Device string ("cpu", "0", "0,1", …).
    """
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    # Capture the Ultralytics-managed val run directory
    kwargs = {
        "project": output_dir,
        "name": "detection_eval",
        "exist_ok": True,
        "plots": True,
        "imgsz": imgsz,       # Match training image size for comparability
        "device": device,
    }
    if data_yaml:
        kwargs["data"] = data_yaml

    logger.info(
        f"Running validation | model={model_path} | data={data_yaml or 'model default'} "
        f"| imgsz={imgsz} | device={device}"
    )

    results = model.val(**kwargs)

    # Determine where Ultralytics wrote its own run artifacts
    val_run_dir = str(results.save_dir) if hasattr(results, 'save_dir') else os.path.join(output_dir, "detection_eval")

    map50    = float(results.box.map50) if hasattr(results.box, 'map50') else None
    map50_95 = float(results.box.map)  if hasattr(results.box, 'map')   else None

    # Save self-documenting metrics file
    metrics = {
        "mAP50":       map50,
        "mAP50_95":    map50_95,
        "model_path":  os.path.abspath(model_path),
        "data_yaml":   data_yaml or "model default",
        "imgsz":       imgsz,
        "device":      str(device),
        "timestamp":   datetime.now().isoformat(),
        "val_run_dir": val_run_dir,   # Link to plots / confusion matrix
    }

    metrics_path = os.path.join(output_dir, "detection_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDetection Results:")
    print(f"  mAP@50    : {map50:.4f}" if map50 is not None else "  mAP@50    : N/A")
    print(f"  mAP@50:95 : {map50_95:.4f}" if map50_95 is not None else "  mAP@50:95 : N/A")
    print(f"  Metrics   : {metrics_path}")
    print(f"  Plots     : {val_run_dir}")   # Inform user where to find plots

    logger.info(f"Evaluation complete. Metrics written to: {metrics_path}")
    logger.info(f"Ultralytics val artifacts (plots, confusion matrix): {val_run_dir}")

    return metrics


def compare_metrics(training_run_dir, eval_metrics_path, map_diff_threshold=0.05):
    """
    Reconciles training results.csv with detection_metrics.json.

    Warns if the best mAP values differ significantly (> map_diff_threshold),
    which would indicate the evaluation used a different model/dataset than training.

    Args:
        training_run_dir:  Path to the yolov8_run/ directory containing results.csv
        eval_metrics_path: Path to detection_metrics.json
        map_diff_threshold: Difference above which a warning is raised.

    Returns:
        dict with 'csv_map50', 'json_map50', 'diff', 'mismatch_detected'
    """
    import csv

    results_csv = os.path.join(training_run_dir, "results.csv")
    if not os.path.exists(results_csv):
        logger.warning(f"compare_metrics: results.csv not found at {results_csv}")
        return None

    if not os.path.exists(eval_metrics_path):
        logger.warning(f"compare_metrics: detection_metrics.json not found at {eval_metrics_path}")
        return None

    # Read best mAP50 from training CSV
    best_csv_map50 = 0.0
    with open(results_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = next((k for k in row if 'mAP50' in k and '95' not in k), None)
            if key:
                try:
                    val = float(row[key].strip())
                    best_csv_map50 = max(best_csv_map50, val)
                except ValueError:
                    pass

    # Read mAP50 from eval JSON
    with open(eval_metrics_path) as f:
        eval_data = json.load(f)
    json_map50 = eval_data.get("mAP50")

    if json_map50 is None:
        logger.warning("compare_metrics: 'mAP50' missing from detection_metrics.json")
        return None

    diff = abs(best_csv_map50 - json_map50)
    mismatch = diff > map_diff_threshold

    result = {
        "csv_best_map50":   round(best_csv_map50, 4),
        "json_map50":       round(json_map50, 4),
        "diff":             round(diff, 4),
        "mismatch_detected": mismatch,
    }

    if mismatch:
        logger.warning(
            f"⚠️  METRICS MISMATCH DETECTED\n"
            f"  results.csv best mAP50 : {best_csv_map50:.4f}\n"
            f"  detection_metrics.json : {json_map50:.4f}\n"
            f"  Difference             : {diff:.4f}  (threshold={map_diff_threshold})\n"
            f"  → Evaluation was likely run on a DIFFERENT model, dataset, or imgsz\n"
            f"    than what produced results.csv.\n"
            f"    Check 'model_path', 'data_yaml', 'imgsz' in detection_metrics.json."
        )
    else:
        logger.info(
            f"Metrics reconciliation OK | CSV best mAP50={best_csv_map50:.4f} | "
            f"JSON mAP50={json_map50:.4f} | diff={diff:.4f}"
        )

    return result


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates Grad-CAM heatmap for model interpretability."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, default="detection", choices=["classification", "detection"])
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (must match training imgsz)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, 0, 0,1 ...")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to yolov8_run/ dir to reconcile results.csv with detection_metrics.json")
    args = parser.parse_args()

    if args.task == "detection":
        metrics = evaluate_detection(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=args.device,
        )
        if args.compare:
            compare_metrics(
                training_run_dir=args.compare,
                eval_metrics_path="outputs/evaluation/detection_metrics.json",
            )
