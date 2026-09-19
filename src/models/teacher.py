"""Teacher 1D-CNN model architecture for ECG window classification.

Higher-capacity reference model for 200-sample ECG window arrhythmia detection.
Governed by PRD.md (FR-5, FR-6), Architecture.md §5.3, AGENTS.md §8, and DECISIONS.md #4.
"""

from typing import Optional, Tuple
import tensorflow as tf

from src.pipeline.config import setup_logging

logger = setup_logging()


def build_teacher_model(
    input_shape: Tuple[int, int] = (200, 1),
    num_classes: int = 2,
    learning_rate: float = 1e-3,
    seed: Optional[int] = 42,
) -> tf.keras.Model:
    """Build and compile the Teacher 1D-CNN reference model.

    Args:
        input_shape: Input ECG window shape tuple (samples, channels), default (200, 1).
        num_classes: Number of output classification classes (default 2 for normal/anomaly).
        learning_rate: Adam optimizer learning rate (default 1e-3).
        seed: Optional random seed for reproducible weight initialization.

    Returns:
        Compiled tf.keras.Model instance.
    """
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.layers.Input(shape=input_shape, name="ecg_input")

    # Conv Block 1
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=7, padding="same", activation="relu", name="conv1_1"
    )(inputs)
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=7, padding="same", activation="relu", name="conv1_2"
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # Conv Block 2
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=5, padding="same", activation="relu", name="conv2_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=5, padding="same", activation="relu", name="conv2_2"
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # Conv Block 3
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, padding="same", activation="relu", name="conv3_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, padding="same", activation="relu", name="conv3_2"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Dense Classification Head
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout1")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output_softmax"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Teacher_1D_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(
        f"Teacher 1D-CNN built successfully with {model.count_params():,} parameters."
    )
    return model


if __name__ == "__main__":
    teacher_model = build_teacher_model()
    teacher_model.summary()
