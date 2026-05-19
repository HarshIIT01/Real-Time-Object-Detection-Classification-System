"""
Test suite for the Real-Time Object Detection & Classification System.
"""
import pytest
import os
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  UTILS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestUtils:
    def test_load_config(self):
        from src.utils import load_config
        config = load_config("configs/config.yaml")
        assert 'system' in config
        assert 'data' in config
        assert 'model' in config
        assert 'training' in config
        assert config['system']['task'] in ('classification', 'detection')

    def test_load_config_missing(self):
        from src.utils import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_set_seed(self):
        from src.utils import set_seed
        set_seed(42)
        import tensorflow as tf
        a = tf.random.normal([3]).numpy()
        set_seed(42)
        b = tf.random.normal([3]).numpy()
        np.testing.assert_array_equal(a, b)

    def test_get_device_info(self):
        from src.utils import get_device_info
        info = get_device_info()
        assert 'num_gpus' in info
        assert 'tf_version' in info


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestModel:
    @pytest.fixture
    def config(self):
        return {
            'data': {'image_size': [224, 224]},
            'model': {
                'backbone': 'MobileNetV2',
                'weights': None,
                'freeze_base': True,
                'num_classes': 10,
                'dropout_rate': 0.3,
                'l2_reg': 0.001,
                'head_hidden_units': [256, 128],
            },
        }

    def test_model_build(self, config):
        from src.model import build_model
        model = build_model(config)
        assert model is not None
        assert model.output_shape == (None, 10)

    def test_model_inference(self, config):
        import tensorflow as tf
        from src.model import build_model
        model = build_model(config)
        dummy = tf.random.normal([2, 224, 224, 3])
        out = model(dummy)
        assert out.shape == (2, 10)
        # Softmax outputs should sum to ~1
        sums = tf.reduce_sum(out, axis=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_unsupported_backbone(self, config):
        from src.model import build_model
        config['model']['backbone'] = 'NonExistentNet'
        with pytest.raises(ValueError):
            build_model(config)

    def test_unfreeze_model(self, config):
        from src.model import build_model, unfreeze_model
        model = build_model(config)
        model = unfreeze_model(model, num_layers=20)
        # Should have some trainable weights now
        trainable = sum(1 for l in model.layers if l.trainable)
        assert trainable > 0

    def test_get_preprocess_fn(self):
        from src.model import get_preprocess_fn
        fn = get_preprocess_fn("MobileNetV2")
        assert callable(fn)
        with pytest.raises(ValueError):
            get_preprocess_fn("FakeNet")


# ═══════════════════════════════════════════════════════════════════════════
#  AUGMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAugmentations:
    def test_augmentation_pipeline(self):
        from src.augmentations import get_training_augmentation
        aug = get_training_augmentation((224, 224), strength="medium")
        assert aug is not None

    def test_augmentation_output_shape(self):
        import tensorflow as tf
        from src.augmentations import get_training_augmentation
        aug = get_training_augmentation((224, 224))
        dummy = tf.random.normal([4, 224, 224, 3])
        out = aug(dummy, training=True)
        assert out.shape == (4, 224, 224, 3)

    def test_mixup(self):
        import tensorflow as tf
        from src.augmentations import mixup
        imgs = tf.random.normal([8, 32, 32, 3])
        labels = tf.one_hot(tf.range(8) % 3, 3)
        mixed_imgs, mixed_labels = mixup(imgs, labels)
        assert mixed_imgs.shape == imgs.shape
        assert mixed_labels.shape == labels.shape


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetManager:
    def test_builtin_dataset(self):
        from src.dataset_manager import prepare_detection_dataset
        config = {'data': {'detection_dataset': 'coco8.yaml', 'dataset_path': 'data/dataset'}}
        result = prepare_detection_dataset(config)
        assert result == 'coco8.yaml'

    def test_roboflow_no_key(self):
        from src.dataset_manager import prepare_detection_dataset
        config = {
            'data': {
                'detection_dataset': 'roboflow',
                'roboflow': {'api_key': 'YOUR_API_KEY', 'workspace': '', 'project': ''},
            }
        }
        with pytest.raises(ValueError):
            prepare_detection_dataset(config)

    def test_missing_local_yaml(self):
        from src.dataset_manager import prepare_detection_dataset
        config = {'data': {'detection_dataset': '/nonexistent/path.yaml'}}
        with pytest.raises(FileNotFoundError):
            prepare_detection_dataset(config)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigValidation:
    def test_valid_config(self):
        from src.utils import load_config
        config = load_config("configs/config.yaml")
        assert config['data']['detection_imgsz'] % 32 == 0

    def test_invalid_task(self):
        from src.utils import _validate_config
        with pytest.raises(ValueError):
            _validate_config({
                'system': {'task': 'segmentation'},
                'data': {}, 'model': {}, 'training': {},
                'inference': {}, 'export': {},
            })
