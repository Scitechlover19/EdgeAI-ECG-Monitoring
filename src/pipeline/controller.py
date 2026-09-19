"""PipelineController module orchestrating the end-to-end EdgeAI ECG monitoring loop.

Wires DataIngestion -> DSPFilter -> TinyMLEngine -> StateScheduler -> SecureTelemetry -> TelemetrySink.
Governed by PRD.md, Architecture.md §2, §4, §5.8, AGENTS.md, and DECISIONS.md.

CRITICAL RULE: Working Vertical Slice execution harness.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.dsp.dsp_filter import DSPFilter
from src.edge.tinyml_engine import TinyMLEngine
from src.edge.virtual_mcu import VirtualMCU
from src.ingestion.data_ingestion import DataIngestion
from src.pipeline.config import PipelineConfig, load_config, setup_logging
from src.scheduler.state_scheduler import StateScheduler
from src.telemetry.secure_telemetry import SecureTelemetry
from src.telemetry.sink import TelemetrySink

logger = setup_logging()


class PipelineControllerError(Exception):
    """Custom exception raised for pipeline execution errors."""
    pass


class PipelineController:
    """Main execution controller orchestrating the 5-stage software pipeline."""

    def __init__(
        self,
        ingestion: DataIngestion,
        config: Optional[PipelineConfig] = None,
        dsp_filter: Optional[DSPFilter] = None,
        virtual_mcu: Optional[VirtualMCU] = None,
        tinyml_engine: Optional[TinyMLEngine] = None,
        state_scheduler: Optional[StateScheduler] = None,
        secure_telemetry: Optional[SecureTelemetry] = None,
        sink: Optional[TelemetrySink] = None,
    ) -> None:
        """Initialize PipelineController with dependencies or defaults.

        Args:
            ingestion: DataIngestion instance for sliding-window feed.
            config: Optional PipelineConfig. Loaded from default config.yaml if None.
            dsp_filter: Optional DSPFilter instance.
            virtual_mcu: Optional VirtualMCU instance.
            tinyml_engine: Optional TinyMLEngine instance.
            state_scheduler: Optional StateScheduler instance.
            secure_telemetry: Optional SecureTelemetry instance.
            sink: Optional TelemetrySink instance.
        """
        self.config = config if config is not None else load_config()
        self.ingestion = ingestion

        self.dsp_filter = (
            dsp_filter
            if dsp_filter is not None
            else DSPFilter(
                dsp_config=self.config.dsp,
                sample_rate=self.ingestion.sample_rate,
            )
        )

        self.virtual_mcu = (
            virtual_mcu
            if virtual_mcu is not None
            else VirtualMCU(
                sram_limit_kb=self.config.budgets.sram_limit_kb,
                flash_limit_mb=self.config.budgets.flash_limit_mb,
                latency_budget_ms=self.config.budgets.latency_limit_ms,
            )
        )

        self.tinyml_engine = (
            tinyml_engine
            if tinyml_engine is not None
            else TinyMLEngine(virtual_mcu=self.virtual_mcu)
        )

        self.sink = sink if sink is not None else TelemetrySink()

        self.secure_telemetry = (
            secure_telemetry
            if secure_telemetry is not None
            else SecureTelemetry(sink=self.sink)
        )

        self.state_scheduler = (
            state_scheduler
            if state_scheduler is not None
            else StateScheduler(
                threshold=self.config.scheduler.threshold,
                telemetry_callback=self.secure_telemetry.transmit_alert,
            )
        )
        # Ensure scheduler is wired to telemetry callback
        if self.state_scheduler.telemetry_callback is None:
            self.state_scheduler.telemetry_callback = self.secure_telemetry.transmit_alert

        self.window_log: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """Execute the end-to-end pipeline loop over all available windows.

        Returns:
            Dict containing detailed summary execution statistics.
        """
        self.window_log.clear()
        latencies_ms: List[float] = []

        logger.info(
            f"Starting pipeline execution: {self.ingestion.total_windows} windows, sample rate={self.ingestion.sample_rate} Hz"
        )

        for w_idx, t_sec, raw_window in self.ingestion.stream_windows():
            # Step 1: DSP Preprocessing (Bandpass + Baseline + Normalization)
            clean_window, dsp_latency = self.virtual_mcu.profile_execution(
                self.dsp_filter.process_window, raw_window
            )

            # Step 2: TinyML Inference
            confidence, label_idx, label_str, infer_latency = self.tinyml_engine.invoke_inference(clean_window)
            per_window_latency = dsp_latency + infer_latency
            latencies_ms.append(per_window_latency)

            # Step 3 & 4: StateScheduler Decision & SecureTelemetry (only if confidence >= threshold)
            final_state, triggered, tel_response = self.state_scheduler.process_window_result(
                window_index=w_idx,
                timestamp_sec=t_sec,
                confidence_score=confidence,
                anomaly_id=label_idx if label_idx != 0 else 1,
            )

            window_record = {
                "window_index": w_idx,
                "timestamp_sec": round(t_sec, 3),
                "confidence_score": round(confidence, 4),
                "predicted_label": label_str,
                "telemetry_triggered": triggered,
                "resulting_state": final_state,
                "dsp_latency_ms": round(dsp_latency, 3),
                "inference_latency_ms": round(infer_latency, 3),
                "total_latency_ms": round(per_window_latency, 3),
            }
            self.window_log.append(window_record)

        # Aggregate Execution Summary
        total_processed = len(self.window_log)
        anomaly_triggers = len([w for w in self.window_log if w["telemetry_triggered"]])

        mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        summary = {
            "total_windows": total_processed,
            "anomaly_triggers": anomaly_triggers,
            "radio_off_windows": total_processed - anomaly_triggers,
            "telemetry_records_received": len(self.sink.get_records()),
            "mean_latency_ms": round(mean_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "latency_budget_ms": self.config.budgets.latency_limit_ms,
            "latency_budget_met": mean_latency <= self.config.budgets.latency_limit_ms,
            "status_label": "[SIMULATED]",
        }

        logger.info(
            f"Pipeline complete! Processed {total_processed} windows | Alerts sent: {anomaly_triggers} | Mean latency: {mean_latency:.2f} ms"
        )
        return summary
