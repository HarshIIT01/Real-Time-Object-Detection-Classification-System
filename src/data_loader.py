"""
Data Loader with advanced preprocessing and augmentation pipelines.
Supports classification via tf.data and delegates detection to Ultralytics.
"""
import tensorflow as tf
import os
import logging

from src.augmentations import get_training_augmentation, mixup

logger = logging.getLogger("RT_ObjectDetection")

class DataLoader:
    """Efficient tf.data-based data loader for image classification."""

    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data']['dataset_path']
        self.image_size = tuple(config['data']['image_size'])
        self.batch_size = config['data']['batch_size']
        self.autotune = tf.data.AUTOTUNE if config['data'].get('autotune', True) else None
        self.seed = config['system']['seed']
        self.class_names = None

    def create_dataset_from_directory(self):
        """
        Creates optimized tf.data.Dataset pipelines from directory structure.

        Expected structure:
            data/dataset/train/<class_name>/<images>
            data/dataset/test/<class_name>/<images>   (optional)

        Returns:
            (train_ds, val_ds, class_names)
        """
        train_path = os.path.join(self.dataset_path, "train")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training data directory not found: {train_path}\n"
                f"Expected structure: {self.dataset_path}/train/<class_name>/<images>"
            )

        val_split = self.config['data'].get('validation_split', 0.2)

        # Training split
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            validation_split=val_split,
            subset="training",
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=True,
        )

        # Validation split
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            validation_split=val_split,
            subset="validation",
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=False,
        )

        self.class_names = train_ds.class_names
        num_classes = len(self.class_names)
        logger.info(f"Found {num_classes} classes: {self.class_names}")
        logger.info(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
        logger.info(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")

        # Update num_classes in config dynamically
        self.config['model']['num_classes'] = num_classes

        # Augmentation pipeline
        augmentation = get_training_augmentation(self.image_size, strength="medium")

        # Apply augmentation to training data
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=self.autotune,
        )

        # Optional MixUp augmentation (20% of the time for regularization)
        use_mixup = self.config.get('training', {}).get('use_mixup', False)
        if use_mixup:
            logger.info("MixUp augmentation enabled.")
            train_ds = train_ds.map(
                lambda x, y: mixup(x, y, alpha=0.2),
                num_parallel_calls=self.autotune,
            )

        # Performance optimizations
        train_ds = train_ds.cache().prefetch(buffer_size=self.autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=self.autotune)

        return train_ds, val_ds, self.class_names

    def create_test_dataset(self):
        """Creates a test dataset if a test/ directory exists."""
        test_path = os.path.join(self.dataset_path, "test")
        if not os.path.exists(test_path):
            logger.warning(f"No test directory found at {test_path}")
            return None, None

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=False,
        )
        class_names = test_ds.class_names
        test_ds = test_ds.cache().prefetch(buffer_size=self.autotune)
        return test_ds, class_names

    def get_representative_dataset(self, num_samples=200):
        """
        Generator function for TFLite INT8 quantization calibration.
        Uses real validation data instead of random noise.
        """
        train_path = os.path.join(self.dataset_path, "train")
        if not os.path.exists(train_path):
            # Fallback to dummy data
            def dummy_gen():
                for _ in range(num_samples):
                    yield [tf.random.normal([1, *self.image_size, 3], dtype=tf.float32)]
            return dummy_gen

        ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=self.image_size,
            batch_size=1,
            label_mode=None,
            shuffle=True,
            seed=self.seed,
        )

        def representative_gen():
            count = 0
            for img_batch in ds:
                if count >= num_samples:
                    break
                img = tf.cast(img_batch, tf.float32)
                from src.model import get_preprocess_fn
                backbone = self.config['model'].get('backbone', 'MobileNetV2')
                preprocess_fn = get_preprocess_fn(backbone)
                img = preprocess_fn(img)
                yield [img]
                count += 1

        return representative_gen
