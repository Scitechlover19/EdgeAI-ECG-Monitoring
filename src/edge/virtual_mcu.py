"""VirtualMCU module for simulated hardware constraints and resource profiling.

Enforces SRAM limit (256 KB) and Flash limit (1 MB) for model execution.
Tracks simulated tensor arena memory, model memory, and inference latency.
Governed by PRD.md (NFR-1, NFR-2, NFR-3), Architecture.md §5.7 & §10, AGENTS.md Rule 4,
and DECISIONS.md #11, #12.

CRITICAL RULE (AGENTS.md §1.4):
Process RSS / host PC RAM is NEVER used as MCU SRAM.
All numbers reported are labeled [SIMULATED] or [ESTIMATED].
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple


class VirtualMCUOverflowError(Exception):
    """Custom exception raised when simulated MCU SRAM or Flash memory limits are breached."""
    pass


class VirtualMCU:
    """Simulated Microcontroller execution harness enforcing resource ceilings."""

    def __init__(
        self,
        sram_limit_kb: float = 256.0,
        flash_limit_mb: float = 1.0,
        latency_budget_ms: float = 50.0,
    ) -> None:
        """Initialize VirtualMCU with hardware resource ceilings.

        Args:
            sram_limit_kb: Maximum simulated SRAM ceiling in KB (default 256.0 KB).
            flash_limit_mb: Maximum simulated Flash ceiling in MB (default 1.0 MB).
            latency_budget_ms: Target per-window latency budget in ms (default 50.0 ms).
        """
        self.sram_limit_kb = float(sram_limit_kb)
        self.flash_limit_mb = float(flash_limit_mb)
        self.latency_budget_ms = float(latency_budget_ms)

        self.sram_limit_bytes = int(self.sram_limit_kb * 1024)
        self.flash_limit_bytes = int(self.flash_limit_mb * 1024 * 1024)

    def calculate_tensor_arena(
        self,
        model_size_bytes: int,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        working_buffer_bytes: int = 4096,
        dtype_size_bytes: int = 1,  # INT8 default = 1 byte
    ) -> Dict[str, Any]:
        """Estimate peak SRAM utilization for model execution.

        Calculates model parameter size in memory + input/output tensor buffers +
        intermediate working scratchpad memory.

        Args:
            model_size_bytes: Size of the model binary flatbuffer in bytes.
            input_shape: Shape of the input tensor (e.g. (1, 200, 1)).
            output_shape: Shape of the output tensor (e.g. (1, 2)).
            working_buffer_bytes: Additional activation scratchpad memory buffer in bytes.
            dtype_size_bytes: Bytes per element (1 for INT8, 4 for Float32).

        Returns:
            Dict containing detailed memory breakdown labeled with [SIMULATED].

        Raises:
            VirtualMCUOverflowError: If estimated peak SRAM exceeds sram_limit_kb.
        """
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim

        output_elements = 1
        for dim in output_shape:
            output_elements *= dim

        input_tensor_bytes = input_elements * dtype_size_bytes
        output_tensor_bytes = output_elements * dtype_size_bytes

        # Tensor arena estimation: input buffer + output buffer + working activation scratchpad
        estimated_tensor_arena_bytes = input_tensor_bytes + output_tensor_bytes + working_buffer_bytes

        # Total simulated SRAM required = model memory resident in SRAM (or flash-mapped) + tensor arena
        total_sram_bytes = model_size_bytes + estimated_tensor_arena_bytes
        total_sram_kb = total_sram_bytes / 1024.0

        headroom_kb = self.sram_limit_kb - total_sram_kb

        report = {
            "status_label": "[SIMULATED]",
            "model_memory_bytes": model_size_bytes,
            "input_tensor_bytes": input_tensor_bytes,
            "output_tensor_bytes": output_tensor_bytes,
            "working_buffer_bytes": working_buffer_bytes,
            "estimated_tensor_arena_bytes": estimated_tensor_arena_bytes,
            "total_estimated_sram_bytes": total_sram_bytes,
            "total_estimated_sram_kb": round(total_sram_kb, 2),
            "sram_limit_kb": self.sram_limit_kb,
            "headroom_kb": round(headroom_kb, 2),
            "exceeds_limit": total_sram_kb > self.sram_limit_kb,
        }

        if report["exceeds_limit"]:
            raise VirtualMCUOverflowError(
                f"[SIMULATED] SRAM limit exceeded! Required: {total_sram_kb:.2f} KB, Ceiling: {self.sram_limit_kb:.2f} KB"
            )

        return report

    def check_flash_footprint(self, model_file_bytes: int, firmware_overhead_bytes: int = 0) -> Dict[str, Any]:
        """Check deployable Flash footprint against the 1 MB ceiling.

        Args:
            model_file_bytes: Byte size of `.tflite` model file.
            firmware_overhead_bytes: Byte size of compiled C++ harness if present.

        Returns:
            Dict containing Flash report labeled with [SIMULATED].

        Raises:
            VirtualMCUOverflowError: If total flash footprint exceeds flash_limit_mb.
        """
        total_flash_bytes = model_file_bytes + firmware_overhead_bytes
        total_flash_mb = total_flash_bytes / (1024.0 * 1024.0)
        headroom_mb = self.flash_limit_mb - total_flash_mb

        report = {
            "status_label": "[SIMULATED]",
            "model_size_bytes": model_file_bytes,
            "firmware_overhead_bytes": firmware_overhead_bytes,
            "total_flash_bytes": total_flash_bytes,
            "total_flash_mb": round(total_flash_mb, 4),
            "flash_limit_mb": self.flash_limit_mb,
            "headroom_mb": round(headroom_mb, 4),
            "exceeds_limit": total_flash_mb > self.flash_limit_mb,
        }

        if report["exceeds_limit"]:
            raise VirtualMCUOverflowError(
                f"[SIMULATED] Flash limit exceeded! Size: {total_flash_mb:.4f} MB, Ceiling: {self.flash_limit_mb:.4f} MB"
            )

        return report

    def profile_execution(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        """Execute callable and measure high-precision wall-clock latency using time.perf_counter().

        Args:
            func: Function to benchmark (e.g. DSP filter or model inference).
            *args: Arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Tuple of (function output, latency_in_milliseconds).
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        return result, latency_ms
