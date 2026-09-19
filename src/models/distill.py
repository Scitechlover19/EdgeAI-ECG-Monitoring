"""Knowledge Distillation module for Teacher-to-Student model distillation.

Implements soft-target knowledge distillation with temperature scaling T=3.0,
alpha=0.7 weighting, and frozen Teacher parameters.
Governed by PRD.md (FR-7, FR-8), Architecture.md §5.3, AGENTS.md §8, and DECISIONS.md #6.
"""

from typing import Any, Dict, Optional, Tuple
import tensorflow as tf

from src.pipeline.config import setup_logging

logger = setup_logging()


class Distiller(tf.keras.Model):
    """Distiller Keras wrapper for training a Student model using Teacher soft labels."""

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.7,
    ) -> None:
        """Initialize Distiller instance.

        Args:
            student: Student Keras model instance.
            teacher: Teacher Keras model instance.
            temperature: Distillation temperature T (default 3.0).
            alpha: Weight factor for soft-label loss vs hard-label loss (default 0.7).
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha = float(alpha)

        # Freeze Teacher parameters explicitly
        self.teacher.trainable = False

        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.distill_loss_fn = tf.keras.losses.KLDivergence()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self) -> list:
        """Return tracked distillation metrics."""
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.accuracy_tracker,
        ]

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        student_loss_fn: Optional[tf.keras.losses.Loss] = None,
        distill_loss_fn: Optional[tf.keras.losses.Loss] = None,
        metrics: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the distiller with optimizer and custom loss functions."""
        super().compile(optimizer=optimizer, **kwargs)
        if student_loss_fn is not None:
            self.student_loss_fn = student_loss_fn
        if distill_loss_fn is not None:
            self.distill_loss_fn = distill_loss_fn

    def _compute_soft_targets(self, probabilities: tf.Tensor) -> tf.Tensor:
        """Apply temperature scaling T to probability outputs.

        Args:
            probabilities: Output probabilities tensor.

        Returns:
            Temperature-softened probability distribution.
        """
        eps = 1e-7
        safe_probs = tf.maximum(probabilities, eps)
        scaled_logits = tf.math.log(safe_probs) / self.temperature
        return tf.nn.softmax(scaled_logits, axis=-1)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step implementing soft-target distillation loss.

        Args:
            data: Tuple of (x_inputs, y_labels).

        Returns:
            Dictionary of updated metric values.
        """
        x, y = data

        # Ensure teacher remains frozen
        self.teacher.trainable = False

        # Forward pass Teacher (no gradient tracking)
        teacher_predictions = self.teacher(x, training=False)

        # Forward pass Student (with gradient tracking)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            # 1. Hard-target loss (sparse cross-entropy against true labels y)
            student_loss = self.student_loss_fn(y, student_predictions)

            # 2. Soft-target loss (KL divergence between temperature-scaled predictions)
            teacher_soft = self._compute_soft_targets(teacher_predictions)
            student_soft = self._compute_soft_targets(student_predictions)
            distill_loss = self.distill_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)

            # 3. Combined loss blend per DECISIONS.md #6
            total_loss = (1.0 - self.alpha) * student_loss + self.alpha * distill_loss

        # Compute gradients wrt student trainable variables ONLY
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Apply gradients to update student weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update tracking metrics
        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.accuracy_tracker.update_state(y, student_predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom validation/evaluation step.

        Args:
            data: Tuple of (x_inputs, y_labels).

        Returns:
            Dictionary of evaluation metric values.
        """
        x, y = data

        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, student_predictions)

        teacher_soft = self._compute_soft_targets(teacher_predictions)
        student_soft = self._compute_soft_targets(student_predictions)
        distill_loss = self.distill_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)

        total_loss = (1.0 - self.alpha) * student_loss + self.alpha * distill_loss

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.accuracy_tracker.update_state(y, student_predictions)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass delegates directly to Student model."""
        return self.student(inputs, training=training)


def create_distiller(
    teacher_model: tf.keras.Model,
    student_model: tf.keras.Model,
    temperature: float = 3.0,
    alpha: float = 0.7,
    learning_rate: float = 1e-3,
) -> Distiller:
    """Factory function to build and compile a Distiller instance.

    Args:
        teacher_model: Teacher Keras model.
        student_model: Student Keras model.
        temperature: Distillation temperature T (default 3.0).
        alpha: Weight for soft-label loss vs hard-label loss (default 0.7).
        learning_rate: Optimizer learning rate (default 1e-3).

    Returns:
        Compiled Distiller instance.
    """
    distiller = Distiller(
        student=student_model,
        teacher=teacher_model,
        temperature=temperature,
        alpha=alpha,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    distiller.compile(optimizer=optimizer)
    logger.info(
        f"Distiller created successfully (Temperature T={temperature}, Alpha={alpha})."
    )
    return distiller
