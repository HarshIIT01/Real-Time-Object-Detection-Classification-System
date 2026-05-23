"""
Training pipeline for both Classification (TF/Keras) and Detection (YOLOv8).

Key accuracy improvements:
  - Cosine annealing LR schedule
  - Progressive unfreezing with frozen BatchNorm
  - Warmup epochs
  - YOLOv8 trained at 640px with mosaic/mixup/copy-paste augmentation
  - All YOLO hyperparameters exposed via config.yaml
"""
import os
import argparse
import json
import logging

import tensorflow as tf

from src.utils import load_config, setup_logging, set_seed, enable_mixed_precision, save_training_metadata
from src.data_loader import DataLoader
from src.model import build_model, unfreeze_model

logger = setup_logging()

# ── Optional W&B ──
try:
    import wandb
    from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _get_optimizer(config, learning_rate):
    """Creates an optimizer based on config."""
    name = config['training'].get('optimizer', 'adamw').lower()
    weight_decay = config['training'].get('weight_decay', 0.0005)

    if name == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    else:  # adam
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _get_lr_schedule(config, steps_per_epoch):
    """Creates a learning rate schedule."""
    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    use_cosine = config['training'].get('cosine_decay', True)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    if use_cosine:
        # Warmup + Cosine Decay
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1e-7,
            decay_steps=warmup_steps,
            end_learning_rate=lr,
            power=1.0,
        )
        cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=total_steps - warmup_steps,
            alpha=1e-6,
        )

        # Use a registered/serializable LR schedule so inference can load the model.
        from src.lr_schedules import WarmupCosineSchedule

        return WarmupCosineSchedule(warmup_schedule, cosine_schedule, warmup_steps)
    else:
        return lr  # Static LR, rely on ReduceLROnPlateau



def _get_callbacks(config, class_names=None):
    """Creates training callbacks."""
    # Use a classification-specific subdirectory so YOLO and TF
    # artifacts do not collide under outputs/models/
    export_dir = os.path.join(config['export']['export_dir'], 'classification')
    os.makedirs(export_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(export_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.get('logging', {}).get('log_dir', 'logs') + '/tensorboard',
            histogram_freq=1,
        ),
    ]

    # Only add ReduceLROnPlateau if NOT using cosine schedule
    if not config['training'].get('cosine_decay', True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=config['training']['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1,
            )
        )

    if WANDB_AVAILABLE and config.get('logging', {}).get('wandb', False):
        callbacks.extend([
            WandbMetricsLogger(),
            WandbModelCheckpoint(
                filepath=os.path.join(export_dir, 'wandb_best.keras')
            ),
        ])

    # Save class names alongside model
    if class_names:
        class_file = os.path.join(export_dir, 'class_names.json')
        with open(class_file, 'w') as f:
            json.dump(class_names, f)
        logger.info(f"Class names saved to {class_file}")

    return callbacks


def train_classification(config):
    """Full classification training with two-phase transfer learning."""
    set_seed(config['system']['seed'])

    if config['system']['mixed_precision']:
        enable_mixed_precision()

    # Use classification-specific subdir
    clf_export_dir = os.path.join(config['export']['export_dir'], 'classification')
    os.makedirs(clf_export_dir, exist_ok=True)
    # Temporarily inject subdir so helpers pick it up
    config['export']['_clf_export_dir'] = clf_export_dir

    if WANDB_AVAILABLE and config.get('logging', {}).get('wandb', False):
        wandb.init(project="real-time-object-detection", config=config)

    # ── Data ──
    logger.info("Loading dataset...")
    try:
        data_loader = DataLoader(config)
        train_ds, val_ds, class_names = data_loader.create_dataset_from_directory()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.warning(
            "Dataset not found. Create directory structure:\n"
            "  data/dataset/train/<class_name>/<images>\n"
            "Exiting gracefully."
        )
        return

    # ── Model ──
    logger.info("Building model...")
    model = build_model(config)

    # ── Phase 1: Train head only ──
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    lr_schedule = _get_lr_schedule(config, max(steps_per_epoch, 1))

    optimizer = _get_optimizer(config, lr_schedule)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config['model'].get('label_smoothing', 0.1),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    callbacks = _get_callbacks(config, class_names)

    logger.info("═" * 60)
    logger.info("PHASE 1: Training classification head (backbone frozen)")
    logger.info("═" * 60)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
    )

    # ── Phase 2: Fine-tune backbone ──
    logger.info("═" * 60)
    logger.info("PHASE 2: Fine-tuning backbone (progressive unfreezing)")
    logger.info("═" * 60)

    model = unfreeze_model(model, num_layers=config['training']['unfreeze_layers'])

    fine_tune_lr = config['training']['fine_tune_lr']
    fine_tune_epochs = config['training']['fine_tune_epochs']

    # Use lower LR for fine-tuning
    if config['training'].get('cosine_decay', True):
        ft_steps = fine_tune_epochs * max(steps_per_epoch, 1)
        ft_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=fine_tune_lr,
            decay_steps=ft_steps,
            alpha=1e-7,
        )
    else:
        ft_schedule = fine_tune_lr

    fine_tune_optimizer = _get_optimizer(config, ft_schedule)

    model.compile(
        optimizer=fine_tune_optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
    )

    # ── Save final metrics ──
    final_metrics = {
        "phase1_val_accuracy": float(max(history1.history.get('val_accuracy', [0]))),
        "phase2_val_accuracy": float(max(history2.history.get('val_accuracy', [0]))),
        "phase1_val_loss": float(min(history1.history.get('val_loss', [999]))),
        "phase2_val_loss": float(min(history2.history.get('val_loss', [999]))),
    }
    save_training_metadata(config, final_metrics, clf_export_dir)
    logger.info(f"Training complete. Best val accuracy: {final_metrics['phase2_val_accuracy']:.4f}")
    logger.info(f"Classification artifacts saved to: {clf_export_dir}")


    if WANDB_AVAILABLE and config.get('logging', {}).get('wandb', False):
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  DETECTION TRAINING (YOLOv8)
# ═══════════════════════════════════════════════════════════════════════════

def train_detection(config):
    """
    YOLOv8 detection training with all hyperparameters from config.

    Key accuracy improvements over baseline:
      - Proper image size (640 instead of 224)
      - Configurable model size (n/s/m/l/x)
      - Cosine LR schedule
      - Mosaic + MixUp + Copy-Paste augmentation
      - More epochs with patience-based early stopping
    """
    from ultralytics import YOLO
    from src.dataset_manager import prepare_detection_dataset

    logger.info("═" * 60)
    logger.info("DETECTION TRAINING — YOLOv8")
    logger.info("═" * 60)

    # ── Prepare dataset ──
    try:
        yaml_path = prepare_detection_dataset(config)
    except Exception as e:
        logger.error(f"Failed to prepare detection dataset: {e}")
        return

    # ── Load model ──
    yolo_model = config['model'].get('yolo_model', 'yolov8s.pt')
    logger.info(f"Loading YOLOv8 model: {yolo_model}")
    model = YOLO(yolo_model)

    if WANDB_AVAILABLE and config.get('logging', {}).get('wandb', False):
        wandb.init(project="real-time-object-detection-yolo", config=config)

    # ── Gather hyperparameters from config ──
    det_cfg = config['training'].get('detection', {})
    export_dir = os.path.abspath(config['export']['export_dir'])
    imgsz = config['data'].get('detection_imgsz', 640)
    batch = config['data'].get('detection_batch', 16)
    epochs = config['training'].get('epochs', 100)

    # ── Determine device ──
    device_cfg = config['inference'].get('device', 'auto')
    if device_cfg == 'auto':
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_cfg

    # Warn when training on CPU
    if str(device) == 'cpu':
        logger.warning(
            "\u26a0\ufe0f  TRAINING ON CPU — this will be very slow and may not complete "
            "the intended epoch schedule. For production quality, use a GPU."
        )

    # Assert image size is valid before handing to YOLO
    if not (imgsz >= 320 and imgsz % 32 == 0):
        raise ValueError(
            f"detection_imgsz={imgsz} is invalid. Must be >= 320 and divisible by 32. "
            f"Check configs/config.yaml → data.detection_imgsz"
        )

    # On CPU, force workers=0 to avoid Windows multiprocessing deadlocks
    num_workers = 0 if str(device) == 'cpu' else 8

    # Log augmentation config so every run is fully auditable
    logger.info(
        f"TRAINING CONFIG SUMMARY\n"
        f"  imgsz      : {imgsz}\n"
        f"  batch      : {batch}\n"
        f"  epochs     : {epochs}\n"
        f"  device     : {device}\n"
        f"  cos_lr     : {det_cfg.get('cos_lr', True)}\n"
        f"  mosaic     : {det_cfg.get('mosaic', 1.0)}\n"
        f"  mixup      : {det_cfg.get('mixup', 0.15)}\n"
        f"  copy_paste : {det_cfg.get('copy_paste', 0.1)}\n"
        f"  optimizer  : {det_cfg.get('optimizer', 'auto')}\n"
    )


    # ── Train ──
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=export_dir,
        name="yolov8_run",
        exist_ok=True,
        device=device,
        workers=num_workers,       # Set to 0 on CPU to avoid Windows deadlocks
        # Optimizer & LR
        optimizer=det_cfg.get('optimizer', 'auto'),
        lr0=det_cfg.get('lr0', 0.01),
        lrf=det_cfg.get('lrf', 0.01),
        momentum=det_cfg.get('momentum', 0.937),
        weight_decay=det_cfg.get('weight_decay', 0.0005),
        warmup_epochs=det_cfg.get('warmup_epochs', 3.0),
        warmup_momentum=det_cfg.get('warmup_momentum', 0.8),
        warmup_bias_lr=det_cfg.get('warmup_bias_lr', 0.1),
        cos_lr=det_cfg.get('cos_lr', True),        # Cosine LR decay
        patience=det_cfg.get('patience', 30),
        close_mosaic=det_cfg.get('close_mosaic', 10),
        # Augmentation parameters from config
        hsv_h=det_cfg.get('hsv_h', 0.015),
        hsv_s=det_cfg.get('hsv_s', 0.7),
        hsv_v=det_cfg.get('hsv_v', 0.4),
        degrees=det_cfg.get('degrees', 0.0),
        translate=det_cfg.get('translate', 0.1),
        scale=det_cfg.get('scale', 0.5),
        shear=det_cfg.get('shear', 0.0),
        perspective=det_cfg.get('perspective', 0.0),
        flipud=det_cfg.get('flipud', 0.0),
        fliplr=det_cfg.get('fliplr', 0.5),
        mosaic=det_cfg.get('mosaic', 1.0),
        mixup=det_cfg.get('mixup', 0.15),
        copy_paste=det_cfg.get('copy_paste', 0.1),
        erasing=det_cfg.get('erasing', 0.4),
        # Logging
        plots=config.get('logging', {}).get('plots', True),
        save_period=config.get('logging', {}).get('save_period', -1),
        verbose=config.get('logging', {}).get('verbose', True),
    )

    logger.info(f"Dataset: {yaml_path}")

    # ── Log results ──
    best_model_path = os.path.join(export_dir, "yolov8_run", "weights", "best.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Best model saved: {best_model_path}")
    else:
        logger.warning(f"Expected best model at {best_model_path} — check training output.")

    # Save class names inside the YOLO run dir only (not in root export_dir)
    # Save class names inside the YOLO run directory
    try:
        class_names = list(model.names.values()) if hasattr(model, 'names') else []
        if class_names:
            cn_path = os.path.join(export_dir, "yolov8_run", "class_names.json")
            os.makedirs(os.path.dirname(cn_path), exist_ok=True)
            with open(cn_path, 'w') as f:
                json.dump(class_names, f)
            logger.info(f"Detection class names saved: {cn_path}")
    except Exception:
        pass

    logger.info("Detection training complete.")

    if WANDB_AVAILABLE and config.get('logging', {}).get('wandb', False):
        wandb.finish()

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def train(config_path):
    """Main training dispatcher."""
    config = load_config(config_path)
    task = config['system'].get('task', 'classification')

    logger.info(f"Task: {task.upper()}")
    logger.info(f"Config: {config_path}")

    if task == 'detection':
        train_detection(config)
    else:
        train_classification(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detection or classification model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
