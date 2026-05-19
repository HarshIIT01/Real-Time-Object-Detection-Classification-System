"""
Advanced Data Augmentation Pipelines.
Provides both Keras-layer-based augmentations for classification
and parameter configs for YOLO detection augmentation.
"""
import tensorflow as tf
import numpy as np


def get_training_augmentation(image_size, strength="medium"):
    """
    Returns a tf.keras.Sequential augmentation pipeline.

    Args:
        image_size: Tuple (H, W) — used for context but not resizing.
        strength: "light", "medium", or "heavy".

    Returns:
        A Keras Sequential model that augments image batches.
    """
    configs = {
        "light": {
            "rotation": 0.05,
            "zoom": (-0.1, 0.1),
            "translation": 0.05,
            "contrast": 0.1,
            "brightness": 0.1,
        },
        "medium": {
            "rotation": 0.15,
            "zoom": (-0.2, 0.2),
            "translation": 0.1,
            "contrast": 0.2,
            "brightness": 0.15,
        },
        "heavy": {
            "rotation": 0.3,
            "zoom": (-0.3, 0.3),
            "translation": 0.15,
            "contrast": 0.3,
            "brightness": 0.2,
        },
    }

    cfg = configs.get(strength, configs["medium"])

    data_augmentation = tf.keras.Sequential([
        # Geometric
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(cfg["rotation"]),
        tf.keras.layers.RandomZoom(
            height_factor=cfg["zoom"],
            width_factor=cfg["zoom"],
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=cfg["translation"],
            width_factor=cfg["translation"],
        ),
        # Photometric
        tf.keras.layers.RandomContrast(cfg["contrast"]),
        tf.keras.layers.RandomBrightness(cfg["brightness"]),
    ], name="data_augmentation")

    return data_augmentation


def mixup(images, labels, alpha=0.2):
    """
    MixUp augmentation: blends pairs of images and labels.

    Samples mixing weight from a Beta(alpha, alpha) distribution.
    """
    batch_size = tf.shape(images)[0]

    # Beta distribution via Gamma samples
    weight = tf.random.gamma([batch_size, 1], alpha)
    weight_y = tf.random.gamma([batch_size, 1], alpha)
    lam = weight / (weight + weight_y + 1e-8)

    image_weight = tf.reshape(lam, [batch_size, 1, 1, 1])
    label_weight = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)

    images_mixed = images * image_weight + images_shuffled * (1.0 - image_weight)
    labels_mixed = labels * label_weight + labels_shuffled * (1.0 - label_weight)

    return images_mixed, labels_mixed


def cutmix(images, labels, alpha=1.0):
    """
    CutMix augmentation: cuts and pastes rectangular patches between images.

    Adjusts labels proportionally to the area of the patch.
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    # Sample lambda from Beta distribution
    lam = tf.random.gamma([1], alpha)
    lam_y = tf.random.gamma([1], alpha)
    lam = lam / (lam + lam_y + 1e-8)
    lam = tf.squeeze(lam)

    # Get random bounding box
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)

    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)

    x1 = tf.maximum(cx - cut_w // 2, 0)
    y1 = tf.maximum(cy - cut_h // 2, 0)
    x2 = tf.minimum(cx + cut_w // 2, img_w)
    y2 = tf.minimum(cy + cut_h // 2, img_h)

    # Shuffle images
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)

    # Create mask
    mask = tf.ones_like(images[:, :, :, :1])
    padding = tf.zeros([batch_size, y2 - y1, x2 - x1, tf.shape(images)[3]])

    # Apply CutMix via padding operations
    top = images[:, :y1, :, :]
    mid_left = images[:, y1:y2, :x1, :]
    mid_right = images[:, y1:y2, x2:, :]
    bottom = images[:, y2:, :, :]

    mid_center = images_shuffled[:, y1:y2, x1:x2, :]

    mid = tf.concat([mid_left, mid_center, mid_right], axis=2)
    result = tf.concat([top, mid, bottom], axis=1)

    # Adjust lambda based on actual cut area
    actual_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    total_area = tf.cast(img_h * img_w, tf.float32)
    adjusted_lam = 1.0 - actual_area / total_area

    label_weight = tf.reshape(adjusted_lam, [1, 1])
    labels_mixed = labels * label_weight + labels_shuffled * (1.0 - label_weight)

    return result, labels_mixed


def random_erasing(images, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    """
    Random Erasing augmentation: randomly erases a rectangular region.
    """
    def _erase(image):
        if tf.random.uniform([]) > probability:
            return image

        img_h = tf.shape(image)[0]
        img_w = tf.shape(image)[1]
        img_c = tf.shape(image)[2]
        area = tf.cast(img_h * img_w, tf.float32)

        target_area = tf.random.uniform([], sl, sh) * area
        aspect_ratio = tf.random.uniform([], r1, 1.0 / r1)

        h = tf.cast(tf.math.sqrt(target_area * aspect_ratio), tf.int32)
        w = tf.cast(tf.math.sqrt(target_area / aspect_ratio), tf.int32)

        h = tf.minimum(h, img_h)
        w = tf.minimum(w, img_w)

        y = tf.random.uniform([], 0, img_h - h + 1, dtype=tf.int32)
        x = tf.random.uniform([], 0, img_w - w + 1, dtype=tf.int32)

        noise = tf.random.uniform([h, w, img_c], dtype=image.dtype)

        # Create and apply mask
        top = image[:y, :, :]
        mid_left = image[y:y + h, :x, :]
        mid_right = image[y:y + h, x + w:, :]
        bottom = image[y + h:, :, :]

        mid = tf.concat([mid_left, noise, mid_right], axis=1)
        result = tf.concat([top, mid, bottom], axis=0)

        return result

    return tf.map_fn(_erase, images)
