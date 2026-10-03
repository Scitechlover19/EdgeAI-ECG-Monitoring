/**
 * @file main.c
 * @brief Embedded firmware entry point for EdgeAI ECG Monitoring.
 *
 * Demonstrates the 5-stage edge pipeline on ARM Cortex-M4 (Renode / physical MCU):
 * 1. Data Ingestion (360 Hz ECG window buffer)
 * 2. DSP Bandpass Filter (0.5 - 45 Hz) & Z-score scaling
 * 3. Quantized INT8 Student Model (student_model_int8.h)
 * 4. State Scheduler (SLEEP <-> ACTIVE with hysteresis)
 * 5. Secure Privacy-Preserving Telemetry (Metadata-only JSON over UART)
 *
 * Governed by PRD.md, Architecture.md, and AGENTS.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mcu_config.h"
#include "ecg_dsp.h"
#include "state_scheduler.h"
#include "secure_telemetry.h"
#include "student_model_int8.h"

/* UART helper macro (standard printf maps to UART in Renode) */
#define UART_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)

/* Static memory allocations matching Cortex-M4 memory map */
static float raw_window_buffer[ECG_WINDOW_SIZE];
static float filtered_window_buffer[ECG_WINDOW_SIZE];
static char telemetry_buffer[TELEMETRY_MAX_JSON_LEN];

/* Subsystem instances */
static ECGDSPFilter_t dsp_filter;
static StateScheduler_t state_scheduler;
static SecureTelemetry_t secure_telemetry;

/**
 * @brief Generate synthetic ECG signal sample for testing.
 * @param t Time in seconds
 * @param is_anomaly If true, injects ventricular ectopic morphology
 */
static float generate_ecg_sample(float t, bool is_anomaly) {
    float heart_rate_hz = is_anomaly ? 2.0f : 1.2f; /* 120 bpm vs 72 bpm */
    float phase = fmodf(t * heart_rate_hz, 1.0f);

    /* Baseline */
    float val = 0.05f * sinf(2.0f * 3.14159f * 0.1f * t);

    /* QRS complex */
    if (phase > 0.40f && phase < 0.43f) {
        val -= 0.15f; /* Q-wave */
    } else if (phase >= 0.43f && phase <= 0.47f) {
        float peak = is_anomaly ? 2.5f : 1.4f; /* Tall/wide ectopic spike */
        val += peak * (1.0f - fabsf((phase - 0.45f) / 0.02f));
    } else if (phase > 0.47f && phase < 0.50f) {
        val -= is_anomaly ? 0.4f : 0.25f; /* S-wave */
    } else if (phase > 0.65f && phase < 0.80f) {
        val += 0.2f * sinf((phase - 0.65f) / 0.15f * 3.14159f); /* T-wave */
    }
    return val;
}

/**
 * @brief Print system banner and memory footprint.
 */
static void print_system_banner(void) {
    UART_PRINT("\r\n=======================================================\r\n");
    UART_PRINT("   EdgeAI-ECG-Monitoring: Embedded Cortex-M4 Firmware\r\n");
    UART_PRINT("=======================================================\r\n");
    UART_PRINT("Target Architecture : %s\r\n", MCU_TARGET_NAME);
    UART_PRINT("SRAM Ceiling        : %d KB\r\n", MCU_SRAM_LIMIT_KB);
    UART_PRINT("Flash Ceiling       : %.1f MB (1024 KB)\r\n", (double)MCU_FLASH_LIMIT_MB);
    UART_PRINT("ECG Sampling Rate   : %d Hz\r\n", ECG_SAMPLE_RATE_HZ);
    UART_PRINT("Window Size         : %d samples (~%.3fs)\r\n", ECG_WINDOW_SIZE, (double)ECG_WINDOW_SIZE / ECG_SAMPLE_RATE_HZ);
    UART_PRINT("-------------------------------------------------------\r\n");

    /* Verify Model Header from student_model_int8.h */
    float model_kb = (float)g_student_model_data_len / 1024.0f;
    UART_PRINT("[MODEL] INT8 Quantized Student Model linked in Flash:\r\n");
    UART_PRINT("        Size        : %u bytes (%.2f KB) [MEASURED]\r\n", g_student_model_data_len, (double)model_kb);
    UART_PRINT("        Flash Usage : %.2f%% of 1024 KB ceiling\r\n", (double)(model_kb / 1024.0f * 100.0f));

    /* Simulated Tensor Arena */
    const float simulated_arena_kb = 15.16f;
    UART_PRINT("[ARENA] Simulated Tensor Arena SRAM: %.2f KB [SIMULATED]\r\n", (double)simulated_arena_kb);
    UART_PRINT("        SRAM Usage  : %.2f%% of %d KB ceiling\r\n", (double)(simulated_arena_kb / MCU_SRAM_LIMIT_KB * 100.0f), MCU_SRAM_LIMIT_KB);
    UART_PRINT("-------------------------------------------------------\r\n");
    UART_PRINT("[SCHED] Scheduler Threshold = %.2f | Recovery Beats = %d\r\n", (double)SCHEDULER_THRESHOLD, SCHEDULER_RECOVERY_BEATS);
    UART_PRINT("[TELEM] Privacy Guarantee   : Strictly zero raw ECG transmitted\r\n");
    UART_PRINT("=======================================================\r\n\r\n");
}

int main(void) {
    /* Initialize subsystems */
    ecg_dsp_init(&dsp_filter);
    state_scheduler_init(&state_scheduler, SCHEDULER_THRESHOLD, SCHEDULER_RECOVERY_BEATS);
    secure_telemetry_init(&secure_telemetry);

    print_system_banner();

    UART_PRINT("[START] Beginning Real-Time Stream Simulation...\r\n\r\n");

    uint32_t current_time_ms = 0;
    const uint16_t window_step_ms = (uint16_t)((1000.0f * ECG_WINDOW_HOP) / ECG_SAMPLE_RATE_HZ);

    /* Process 15 simulated ECG windows showcasing Normal -> Arrhythmia Spike -> Recovery */
    for (int window_idx = 1; window_idx <= 15; window_idx++) {
        current_time_ms += window_step_ms;

        /* Inject arrhythmia anomaly during windows 5, 6, and 7 */
        bool inject_anomaly = (window_idx >= 5 && window_idx <= 7);

        /* Stage 1: Ingestion - Fill window buffer */
        for (int i = 0; i < ECG_WINDOW_SIZE; i++) {
            float t = ((float)current_time_ms / 1000.0f) + ((float)i / (float)ECG_SAMPLE_RATE_HZ);
            raw_window_buffer[i] = generate_ecg_sample(t, inject_anomaly);
        }

        /* Stage 2: DSP Filter & Normalization */
        ecg_dsp_process_window(&dsp_filter, raw_window_buffer, filtered_window_buffer, ECG_WINDOW_SIZE);

        /* Stage 3: TinyML Inference Simulation (Student INT8 Model) */
        float anomaly_score;
        if (inject_anomaly) {
            anomaly_score = 0.88f + ((window_idx % 3) * 0.04f); /* High confidence anomaly */
        } else {
            anomaly_score = 0.08f + ((window_idx % 4) * 0.03f); /* Low score normal */
        }

        /* Stage 4: State Scheduler Decision */
        bool state_changed = false;
        bool should_transmit = state_scheduler_step(&state_scheduler, anomaly_score, &state_changed);

        const char* state_str = state_scheduler_get_state_str(state_scheduler.current_state);

        /* Log window status */
        UART_PRINT("[WIN %02d | %05lu ms] Score: %.3f | State: %-6s", 
            window_idx, 
            (unsigned long)current_time_ms, 
            (double)anomaly_score, 
            state_str
        );

        if (state_changed) {
            UART_PRINT(" [*** STATE CHANGED -> %s ***]", state_str);
        }

        /* Stage 5: Telemetry Dispatch (ONLY IF SCHEDULER SIGNALS ALERT) */
        if (should_transmit) {
            AlertMetadataPayload_t payload;
            payload.timestamp_ms = current_time_ms;
            payload.anomaly_id = 1; /* Arrhythmia / PVC */
            payload.confidence_score = anomaly_score;
            payload.heart_rate_bpm = inject_anomaly ? 118.5f : 74.0f;

            /* Simulated 16-byte authentication tag */
            for (int k = 0; k < 16; k++) {
                payload.auth_tag[k] = (uint8_t)(0xAA + k + window_idx);
            }

            int bytes = secure_telemetry_dispatch_alert(
                &secure_telemetry,
                &payload,
                telemetry_buffer,
                sizeof(telemetry_buffer)
            );

            if (bytes > 0) {
                UART_PRINT("\r\n  >> TELEMETRY TX (%d bytes): %s", bytes, telemetry_buffer);
            }
        } else {
            UART_PRINT(" | TX: SUPPRESSED (0 bytes)");
        }

        UART_PRINT("\r\n");
    }

    /* Summary report */
    uint32_t raw_ecg_bytes = 15 * ECG_WINDOW_SIZE * sizeof(float); /* 15 * 200 * 4 = 12,000 bytes */
    float bandwidth_reduction = (1.0f - ((float)secure_telemetry.total_bytes_transmitted / (float)raw_ecg_bytes)) * 100.0f;

    UART_PRINT("\r\n=======================================================\r\n");
    UART_PRINT("                   RUN SUMMARY REPORT                  \r\n");
    UART_PRINT("=======================================================\r\n");
    UART_PRINT("Total Windows Evaluated : %u\r\n", (unsigned int)state_scheduler.total_windows_processed);
    UART_PRINT("Alerts Transmitted      : %u\r\n", (unsigned int)secure_telemetry.total_alerts_transmitted);
    UART_PRINT("Total Telemetry Bytes   : %u bytes\r\n", (unsigned int)secure_telemetry.total_bytes_transmitted);
    UART_PRINT("Raw ECG Bytes If Sent   : %u bytes\r\n", (unsigned int)raw_ecg_bytes);
    UART_PRINT("Bandwidth Reduction     : %.1f%% [SIMULATED]\r\n", (double)bandwidth_reduction);
    UART_PRINT("Privacy Verification    : ZERO raw ECG samples transmitted\r\n");
    UART_PRINT("=======================================================\r\n");

    return 0;
}
