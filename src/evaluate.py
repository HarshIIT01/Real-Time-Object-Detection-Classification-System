"""
Model evaluation for Classification and Detection.
"""
import os
import argparse
import json
import logging
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
    model = tf.keras.models.load_model(model_path)

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


def evaluate_detection(model_path, data_yaml=None, output_dir="outputs/evaluation"):
    """Evaluates YOLOv8 detection model. Reports mAP@50, mAP@50:95."""
    from ultralytics import YOLO
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    kwargs = {"project": output_dir, "name": "detection_eval", "exist_ok": True, "plots": True}
    if data_yaml:
        kwargs["data"] = data_yaml
    results = model.val(**kwargs)

    metrics = {
        "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
        "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else None,
    }
    print("\nDetection Results:", metrics)
    with open(os.path.join(output_dir, "detection_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


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
    args = parser.parse_args()
    if args.task == "detection":
        evaluate_detection(args.model, args.data)
