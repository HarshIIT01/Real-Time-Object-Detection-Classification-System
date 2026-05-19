"""
Classification model builder with multiple backbone support,
advanced custom head, and progressive unfreezing.
"""
import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    EfficientNetB3,
    EfficientNetV2S,
)
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger("RT_ObjectDetection")

# Registry of supported backbones
BACKBONE_REGISTRY = {
    "MobileNetV2": {
        "builder": MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "EfficientNetB0": {
        "builder": EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "EfficientNetB3": {
        "builder": EfficientNetB3,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "EfficientNetV2S": {
        "builder": EfficientNetV2S,
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
    },
}


def build_model(config):
    """
    Builds a classification model with transfer learning.

    Supports multiple backbones and a configurable dense head.
    The head uses: GlobalAveragePooling → BN → Dropout → Dense(s) → Softmax
    """
    image_size = tuple(config['data']['image_size'])
    input_shape = image_size + (3,)
    num_classes = config['model']['num_classes']
    backbone_name = config['model']['backbone']
    dropout_rate = config['model']['dropout_rate']
    l2_reg = config['model']['l2_reg']
    head_units = config['model'].get('head_hidden_units', [512, 256])

    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )

    backbone_info = BACKBONE_REGISTRY[backbone_name]

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Apply backbone-specific preprocessing
    x = backbone_info["preprocess"](inputs)

    # Build backbone
    base_model = backbone_info["builder"](
        input_tensor=x,
        include_top=False,
        weights=config['model']['weights'],
    )

    if config['model']['freeze_base']:
        base_model.trainable = False
        logger.info(f"Backbone '{backbone_name}' frozen ({len(base_model.layers)} layers).")
    else:
        logger.info(f"Backbone '{backbone_name}' fully trainable.")

    # Build custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    for i, units in enumerate(head_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name=f"head_dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"head_bn_{i}")(x)
        # Gradually reduce dropout
        drop = max(dropout_rate - 0.1 * (i + 1), 0.1)
        x = layers.Dropout(drop, name=f"head_drop_{i}")(x)

    # Output — float32 for mixed precision compatibility
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        dtype='float32',
        name="predictions",
    )(x)

    model = Model(inputs, outputs, name=f"{backbone_name}_Classification")

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(
        f"Model built: {total_params:,} total params, "
        f"{trainable_params:,} trainable params, "
        f"{num_classes} output classes."
    )

    return model


def unfreeze_model(model, num_layers=80):
    """
    Progressive unfreezing: unfreezes the top `num_layers` of the base model
    for fine-tuning while keeping BatchNorm layers frozen (critical for
    small datasets to avoid destroying pretrained statistics).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model):
            base_model = layer
            break

    if base_model is None:
        logger.warning("No nested sub-model found. Unfreezing top layers of flat model.")
        for layer in model.layers[-num_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    else:
        base_model.trainable = True
        total = len(base_model.layers)
        frozen = 0
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            frozen += 1
        # Always keep BN frozen in fine-tuning
        for layer in base_model.layers[-num_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        logger.info(
            f"Unfroze {total - frozen} / {total} layers in backbone "
            f"(BatchNorm layers kept frozen)."
        )

    return model


def get_preprocess_fn(backbone_name):
    """Returns the preprocessing function for the specified backbone."""
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return BACKBONE_REGISTRY[backbone_name]["preprocess"]
