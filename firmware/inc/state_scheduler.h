/**
 * @file state_scheduler.h
 * @brief Dynamic two-state power and transmission scheduler.
 *
 * Enforces:
 * - SLEEP (0): low-power duty cycling when rhythm is normal.
 * - ACTIVE (1): alert telemetry active when anomaly confidence exceeds threshold.
 * - Hysteresis recovery: returns to SLEEP after 5 consecutive normal beats.
 *
 * Governed by PRD.md (FR-6) and Architecture.md §5.4.
 */

#ifndef STATE_SCHEDULER_H_
#define STATE_SCHEDULER_H_

#include <stdint.h>
#include <stdbool.h>
#include "mcu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    SystemState_t current_state;
    float threshold;
    uint16_t consecutive_normal_count;
    uint16_t recovery_count_target;
    uint32_t total_windows_processed;
    uint32_t total_active_windows;
    uint32_t state_transition_count;
} StateScheduler_t;

/**
 * @brief Initialize state scheduler with default threshold and recovery limits.
 */
void state_scheduler_init(StateScheduler_t* scheduler, float threshold, uint16_t recovery_target);

/**
 * @brief Evaluate an inference score and update system state.
 * 
 * @param scheduler Pointer to scheduler instance
 * @param anomaly_score Model output anomaly probability [0.0 - 1.0]
 * @param[out] out_state_changed Optional boolean set to true if state transitioned
 * @return bool True if telemetry alert transmission is permitted/required
 */
bool state_scheduler_step(StateScheduler_t* scheduler, float anomaly_score, bool* out_state_changed);

/**
 * @brief Get human-readable state string.
 */
const char* state_scheduler_get_state_str(SystemState_t state);

#ifdef __cplusplus
}
#endif

#endif /* STATE_SCHEDULER_H_ */
