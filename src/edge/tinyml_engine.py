"""TinyMLEngine module for edge inference execution inside VirtualMCU.

Loads quantized `.tflite` models or fallback execution runners, invokes inference per 200-sample window,
returns confidence scores and classification labels, and benchmarks per-window latency.
Governed by PRD.md (FR-9, FR-10), Architecture.md §5.4, and DECISIONS.md #10.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from src.edge.virtual_mcu import VirtualMCU


class TinyMLEngineError(Exception):
    """Custom exception raised for TinyMLEngine failures."""
    pass


class TinyMLEngine:
    """Edge inference engine hosting quantized model execution."""

    LABEL_NORMAL = 0
    LABEL_ANOMALY = 1
    LABEL_NAMES = {LABEL_NORMAL: "NORMAL", LABEL_ANOMALY: "ANOMALY"}

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        virtual_mcu: Optional[VirtualMCU] = None,
        fallback_model_fn: Optional[Callable[[np.ndarray], Tuple[float, int]]] = None,
    ) -> None:
        """Initialize TinyMLEngine with model or fallback runner.

        Args:
            model_path: Optional path to exported .tflite model file.
            virtual_mcu: Optional VirtualMCU instance for SRAM/Flash profiling and latency timing.
            fallback_model_fn: Optional python callable for testing or unquantized vertical slice.

        Raises:
            TinyMLEngineError: If model file cannot be loaded or parameters are invalid.
        """
        self.virtual_mcu = virtual_mcu if virtual_mcu is not None else VirtualMCU()
        self.model_path = Path(model_path) if model_path is not None else None
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.fallback_model_fn = fallback_model_fn

        self.input_details = None
        self.output_details = None
        self.model_bytes: int = 0

        if self.model_path is not None:
            self._load_tflite_model(self.model_path)

    def _load_tflite_model(self, path: Path) -> None:
        """Load .tflite model file into TFLite Interpreter.

        Args:
            path: Path to .tflite flatbuffer.

        Raises:
            TinyMLEngineError: If loading fails or file does not exist.
        """
        if not path.is_file():
            raise TinyMLEngineError(f"TFLite model file not found at {path}")

        try:
            self.model_bytes = path.stat().st_size
            self.interpreter = tf.lite.Interpreter(model_path=str(path))
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Verify hardware bounds via VirtualMCU
            input_shape = tuple(self.input_details[0]["shape"])
            output_shape = tuple(self.output_details[0]["shape"])
            dtype_bytes = 1 if self.input_details[0]["dtype"] in (np.int8, np.uint8) else 4

            self.virtual_mcu.calculate_tensor_arena(
                model_size_bytes=self.model_bytes,
                input_shape=input_shape,
                output_shape=output_shape,
                dtype_size_bytes=dtype_bytes,
            )
            self.virtual_mcu.check_flash_footprint(self.model_bytes)

        except Exception as err:
            raise TinyMLEngineError(f"Failed to load TFLite model '{path}': {err}") from err

    def invoke_inference(self, clean_window: np.ndarray) -> Tuple[float, int, str, float]:
        """Run model inference on a single 200-sample clean window.

        Args:
            clean_window: Preprocessed 1D float32 numpy array of shape (200,).

        Returns:
            Tuple containing:
            - anomaly_confidence (float): Confidence score between 0.0 and 1.0.
            - predicted_label (int): 0 for Normal, 1 for Anomaly.
            - label_str (str): "NORMAL" or "ANOMALY".
            - latency_ms (float): Benchmark execution latency in milliseconds.

        Raises:
            TinyMLEngineError: If window shape is invalid or inference fails.
        """
        if not isinstance(clean_window, np.ndarray) or clean_window.ndim != 1:
            raise TinyMLEngineError("Input clean_window must be a 1D numpy array.")
        if len(clean_window) != 200:
            raise TinyMLEngineError(f"Input clean_window length must be 200, got {len(clean_window)}.")

        def _raw_inference() -> Tuple[float, int]:
            if self.interpreter is not None:
                # Prepare tensor shape (1, 200, 1)
                input_dtype = self.input_details[0]["dtype"]
                input_data = clean_window.reshape(1, 200, 1)

                if input_dtype == np.int8:
                    # Quantize float32 window to INT8 using scale and zero_point
                    scale, zero_point = self.input_details[0]["quantization"]
                    if scale > 0:
                        input_data = np.round(input_data / scale + zero_point).astype(np.int8)
                    else:
                        input_data = input_data.astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)

                self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

                output_dtype = self.output_details[0]["dtype"]
                if output_dtype == np.int8:
                    scale, zero_point = self.output_details[0]["quantization"]
                    if scale > 0:
                        output_data = (output_data.astype(np.float32) - zero_point) * scale

                # Use direct probabilities if output layer is already softmax, else apply softmax
                out_vec = output_data[0]
                if np.all(out_vec >= -1e-3) and np.isclose(np.sum(out_vec), 1.0, atol=0.05):
                    probs = np.clip(out_vec, 0.0, 1.0)
                    s = float(np.sum(probs))
                    if s > 0:
                        probs = probs / s
                else:
                    probs = tf.nn.softmax(out_vec).numpy()

                anomaly_confidence = float(probs[1]) if len(probs) > 1 else float(probs[0])
                predicted_label = int(np.argmax(probs)) if len(probs) > 1 else (1 if anomaly_confidence >= 0.5 else 0)
                return anomaly_confidence, predicted_label

            elif self.fallback_model_fn is not None:
                return self.fallback_model_fn(clean_window)

            else:
                # Baseline heuristic runner for vertical slice testing when model path is not supplied
                # Peak-to-peak amplitude > 3.0 indicates arrhythmia anomaly
                ptp = float(np.ptp(clean_window))
                if ptp <= 3.0:
                    score = max(0.0, min(0.20, (ptp - 1.0) / 10.0))
                    label = self.LABEL_NORMAL
                else:
                    score = min(1.0, 0.5 + (ptp - 3.0) / 5.0)
                    label = self.LABEL_ANOMALY
                return score, label

        (confidence, label_idx), latency_ms = self.virtual_mcu.profile_execution(_raw_inference)
        label_str = self.LABEL_NAMES.get(label_idx, "UNKNOWN")
        return confidence, label_idx, label_str, latency_ms
