import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    """Warmup + Cosine decay LR schedule.

    This class is saved inside the model/optimizer config. To load models
    reliably across processes, it must be registered as a Keras serializable.
    """

    def __init__(self, warmup=None, cosine=None, warmup_steps=0, **kwargs):
        # warmup/cosine may be missing when loading older .keras files.
        # We keep defaults so deserialization can succeed.
        super().__init__()
        self.warmup = warmup
        self.cosine = cosine
        self.warmup_steps = warmup_steps


    def __call__(self, step):
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.warmup(step),
            lambda: self.cosine(step - self.warmup_steps),
        )

    def get_config(self):
        # NOTE: Keras will call from_config() with whatever dict we return here.
        # The optimizer schedule config saved into the .keras file is expected
        # to fully reconstruct this object.
        return {
            "warmup_steps": int(self.warmup_steps),
            # These are serialized via tf/keras get_config mechanism.
            # We rely on the schedules being serializable (they are built-in
            # Keras schedules like PolynomialDecay and CosineDecay).
            "warmup": tf.keras.optimizers.schedules.serialize(self.warmup),
            "cosine": tf.keras.optimizers.schedules.serialize(self.cosine),
        }


