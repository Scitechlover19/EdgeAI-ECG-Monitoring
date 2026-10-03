/**
 * @file mcu_config.h
 * @brief Hardware resource ceilings, window parameters, and telemetry constants.
 *
 * Governed by PRD.md (NFR-1, NFR-2, NFR-3), Architecture.md §5, and AGENTS.md.
 */

#ifndef MCU_CONFIG_H_
#define MCU_CONFIG_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Target Hardware Resource Ceilings */
#define MCU_TARGET_NAME           "ARM Cortex-M4"
#define MCU_SRAM_LIMIT_KB         256
#define MCU_FLASH_LIMIT_MB        1.0f
#define MCU_LATENCY_BUDGET_MS     50.0f

/* Signal Processing Specifications */
#define ECG_SAMPLE_RATE_HZ        360
#define ECG_WINDOW_SIZE           200   /* 200 samples = ~0.556s window */
#define ECG_WINDOW_HOP            100   /* 50% overlap */

/* DSP Filter Cutoffs */
#define DSP_LOW_CUTOFF_HZ         0.5f
#define DSP_HIGH_CUTOFF_HZ        45.0f
#define DSP_FILTER_ORDER          4

/* State Scheduler & Thresholds */
#define SCHEDULER_THRESHOLD       0.35f /* Anomaly trigger threshold */
#define SCHEDULER_RECOVERY_BEATS  5     /* Consecutive normal beats to return to SLEEP */

/* System Power States */
typedef enum {
    SYSTEM_STATE_SLEEP  = 0,  /* Low-power duty-cycled state */
    SYSTEM_STATE_ACTIVE = 1   /* High-alert telemetry active state */
} SystemState_t;

#ifdef __cplusplus
}
#endif

#endif /* MCU_CONFIG_H_ */
