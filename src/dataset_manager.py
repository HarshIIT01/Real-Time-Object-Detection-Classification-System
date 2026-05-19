"""
Dataset Manager — handles downloading and configuring datasets for detection.
Supports COCO, Open Images V7, Roboflow Universe, and local custom datasets.
"""
import os
import logging

logger = logging.getLogger("RT_ObjectDetection")


def prepare_detection_dataset(config):
    """
    Prepares the dataset for YOLOv8 object detection training.

    Returns the path to the YAML dataset config file that Ultralytics expects.

    Supported sources (via config['data']['detection_dataset']):
        - "coco.yaml" / "coco8.yaml" / "coco128.yaml" — Ultralytics built-in
        - "open-images-v7.yaml" — Google Open Images
        - "roboflow" — Download from Roboflow Universe
        - "<path>.yaml" — Local custom YAML file
    """
    dataset_source = config['data'].get('detection_dataset', 'coco8.yaml')

    # ── Built-in Ultralytics datasets ──
    builtin_datasets = {
        'coco.yaml', 'coco8.yaml', 'coco128.yaml',
        'coco-pose.yaml', 'coco8-pose.yaml',
        'coco-seg.yaml', 'coco8-seg.yaml',
        'open-images-v7.yaml',
        'voc.yaml',
        'imagenet.yaml',
        'objects365.yaml',
        'lvis.yaml',
    }

    if dataset_source in builtin_datasets:
        logger.info(
            f"Using built-in Ultralytics dataset: '{dataset_source}'. "
            f"Data will be auto-downloaded on first training run."
        )
        return dataset_source

    # ── Roboflow Universe ──
    if dataset_source == 'roboflow':
        return _download_roboflow_dataset(config)

    # ── Local custom YAML ──
    if os.path.exists(dataset_source):
        logger.info(f"Using local dataset YAML: {dataset_source}")
        _validate_local_yaml(dataset_source)
        return dataset_source

    # ── Not found ──
    raise FileNotFoundError(
        f"Dataset source '{dataset_source}' is not a built-in dataset, "
        f"not 'roboflow', and no local file was found at that path.\n"
        f"Built-in options: {sorted(builtin_datasets)}"
    )


def _download_roboflow_dataset(config):
    """Downloads a dataset from Roboflow Universe in YOLOv8 format."""
    rf_config = config['data'].get('roboflow', {})
    api_key = rf_config.get('api_key', '')
    workspace = rf_config.get('workspace', '')
    project = rf_config.get('project', '')
    version = rf_config.get('version', 1)

    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError(
            "Roboflow API key not configured!\n"
            "Set 'data.roboflow.api_key' in configs/config.yaml.\n"
            "Get your free key at https://app.roboflow.com/settings/api"
        )

    if not workspace or not project:
        raise ValueError(
            "Roboflow workspace and project must be set in config.yaml.\n"
            "Example:\n"
            "  roboflow:\n"
            "    api_key: 'rf_xxxxx'\n"
            "    workspace: 'my-workspace'\n"
            "    project: 'my-project'\n"
            "    version: 1"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "Roboflow package not installed. Run: pip install roboflow"
        )

    logger.info(f"Downloading Roboflow dataset: {workspace}/{project} v{version}...")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    download_dir = os.path.abspath(config['data'].get('dataset_path', 'data/dataset'))
    dataset = proj.version(version).download("yolov8", location=download_dir)

    yaml_path = os.path.join(dataset.location, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Expected data.yaml at {yaml_path} after Roboflow download, but not found."
        )

    logger.info(f"Roboflow dataset downloaded to: {dataset.location}")
    return yaml_path


def _validate_local_yaml(yaml_path):
    """Basic validation of a local dataset YAML file."""
    import yaml as pyyaml
    try:
        with open(yaml_path, 'r') as f:
            data = pyyaml.safe_load(f)

        required_keys = ['train', 'val', 'nc', 'names']
        missing = [k for k in required_keys if k not in data]
        if missing:
            logger.warning(
                f"Dataset YAML '{yaml_path}' is missing keys: {missing}. "
                f"Training may fail."
            )
        else:
            logger.info(
                f"Local dataset validated: {data['nc']} classes, "
                f"names={data['names'][:5]}{'...' if data['nc'] > 5 else ''}"
            )
    except Exception as e:
        logger.warning(f"Could not validate dataset YAML: {e}")
