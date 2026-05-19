"""
Utility functions for the Real-Time Object Detection & Classification System.
Handles configuration loading, logging, seeding, and mixed precision setup.
"""
import yaml
import logging
import os
import sys
import json
from datetime import datetime

import tensorflow as tf
import numpy as np


def load_config(config_path="configs/config.yaml"):
    """Loads configuration from a YAML file with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    _validate_config(config)
    return config


def _validate_config(config):
    """Basic config validation to catch common errors early."""
    required_keys = ['system', 'data', 'model', 'training', 'inference', 'export']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    task = config['system'].get('task', 'classification')
    if task not in ('classification', 'detection'):
        raise ValueError(f"Invalid task '{task}'. Must be 'classification' or 'detection'.")

    if task == 'detection':
        imgsz = config['data'].get('detection_imgsz', 640)
        if imgsz < 32 or imgsz % 32 != 0:
            raise ValueError(f"detection_imgsz must be a multiple of 32, got {imgsz}")


def setup_logging(log_dir="logs", log_file="system.log"):
    """Sets up structured Python logging with console and file outputs."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Prevent duplicate handlers when called multiple times
    logger = logging.getLogger("RT_ObjectDetection")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seed(seed=42):
    """Sets seeds across all libraries for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def enable_mixed_precision():
    """Enables mixed precision training (FP16) for modern GPUs."""
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    logging.getLogger("RT_ObjectDetection").info(
        "Mixed precision (FP16) enabled — compute in float16, variables in float32."
    )


def get_device_info():
    """Returns information about available compute devices."""
    gpus = tf.config.list_physical_devices('GPU')
    info = {
        "num_gpus": len(gpus),
        "gpu_names": [g.name for g in gpus],
        "tf_version": tf.__version__,
        "cuda_available": len(gpus) > 0,
    }
    return info


def save_training_metadata(config, metrics, output_dir):
    """Saves training run metadata for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "device_info": get_device_info(),
        "python_version": sys.version,
    }
    path = os.path.join(output_dir, "training_metadata.json")
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    return path
