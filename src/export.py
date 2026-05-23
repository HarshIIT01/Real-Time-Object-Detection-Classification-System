"""
Model export: TFLite (FP16/INT8), ONNX for classification,
and Ultralytics export for YOLO detection models.
"""
import os
import argparse
import logging
import tensorflow as tf

logger = logging.getLogger("RT_ObjectDetection")


def export_tflite(model_path, output_dir, quantization="fp16", representative_gen=None):
    """Export Keras model to TFLite with optional quantization."""
    print(f"Converting {model_path} → TFLite ({quantization})...")
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        out_name = "model_fp16.tflite"
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_gen:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        out_name = "model_int8.tflite"
    else:
        out_name = "model.tflite"

    tflite_model = converter.convert()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved TFLite model: {out_path} ({size_mb:.1f} MB)")


def export_onnx(model_path, output_dir):
    """Export Keras model to ONNX."""
    try:
        import tf2onnx
        print(f"Converting {model_path} → ONNX...")
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        dynamic_shape = (None,) + input_shape[1:]
        spec = (tf.TensorSpec(dynamic_shape, tf.float32, name="input"),)
        output_path = os.path.join(output_dir, "model.onnx")
        os.makedirs(output_dir, exist_ok=True)
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=output_path)
        print(f"Saved ONNX model: {output_path}")
    except ImportError:
        print("tf2onnx not installed. Run: pip install tf2onnx")


def export_yolo(model_path, fmt="onnx", imgsz=640, half=False, simplify=True, dynamic=True, opset=17):
    """Export YOLOv8 model to various formats."""
    from ultralytics import YOLO
    print(f"Exporting YOLO model: {model_path} → {fmt}")
    model = YOLO(model_path)
    model.export(format=fmt, imgsz=imgsz, half=half, simplify=simplify, dynamic=dynamic, opset=opset)
    print(f"YOLO export complete ({fmt}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/models")
    parser.add_argument("--quantize", type=str, default="fp16", choices=["none", "fp16", "int8"])
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--yolo-format", type=str, default=None, choices=["onnx", "torchscript", "engine", "tflite"])
    args = parser.parse_args()

    ext = os.path.splitext(args.model)[1].lower()
    if ext in ('.pt', '.pth') or args.yolo_format:
        export_yolo(args.model, fmt=args.yolo_format or "onnx")
    else:
        export_tflite(args.model, args.output_dir, args.quantize)
        if args.onnx:
            export_onnx(args.model, args.output_dir)
