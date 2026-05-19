"""
Benchmark tool supporting both TF/Keras and YOLOv8 models.
Reports latency, FPS, and model size.
"""
import time
import os
import argparse
import logging
import numpy as np

logger = logging.getLogger("RT_ObjectDetection")


def benchmark_tf(model_path, input_shape=(1, 224, 224, 3), num_runs=100, warmup_runs=20):
    """Benchmark a TensorFlow/Keras model."""
    import tensorflow as tf

    print(f"Benchmarking TF model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    dummy = np.random.random(input_shape).astype(np.float32)

    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        model.predict(dummy, verbose=0)

    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    for _ in range(num_runs):
        t = time.time()
        model.predict(dummy, verbose=0)
        latencies.append((time.time() - t) * 1000)

    _print_results(latencies, model_path)


def benchmark_yolo(model_path, imgsz=640, num_runs=100, warmup_runs=20):
    """Benchmark a YOLOv8 model."""
    from ultralytics import YOLO

    print(f"Benchmarking YOLO model: {model_path}")
    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        model.predict(dummy, verbose=False)

    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    for _ in range(num_runs):
        t = time.time()
        model.predict(dummy, verbose=False)
        latencies.append((time.time() - t) * 1000)

    _print_results(latencies, model_path)


def _print_results(latencies, model_path):
    """Print benchmark statistics."""
    latencies = np.array(latencies)
    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000 / avg

    size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Model          : {os.path.basename(model_path)}")
    print(f"  Model Size     : {size_mb:.1f} MB")
    print(f"  Avg Latency    : {avg:.2f} ms")
    print(f"  Median Latency : {p50:.2f} ms")
    print(f"  P95 Latency    : {p95:.2f} ms")
    print(f"  P99 Latency    : {p99:.2f} ms")
    print(f"  Estimated FPS  : {fps:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    ext = os.path.splitext(args.model)[1].lower()
    if ext in ('.pt', '.pth', '.engine'):
        benchmark_yolo(args.model, imgsz=args.imgsz, num_runs=args.runs)
    else:
        benchmark_tf(args.model, num_runs=args.runs)
