/**
 * @file state_scheduler.c
 * @brief State scheduler implementation for Cortex-M4 edge node.
 */

#include "state_scheduler.h"

void state_scheduler_init(StateScheduler_t* scheduler, float threshold, uint16_t recovery_target) {
    if (!scheduler) return;
    scheduler->current_state = SYSTEM_STATE_SLEEP;
    scheduler->threshold = threshold > 0.0f ? threshold : SCHEDULER_THRESHOLD;
    scheduler->consecutive_normal_count = 0;
    scheduler->recovery_count_target = recovery_target > 0 ? recovery_target : SCHEDULER_RECOVERY_BEATS;
    scheduler->total_windows_processed = 0;
    scheduler->total_active_windows = 0;
    scheduler->state_transition_count = 0;
}

bool state_scheduler_step(StateScheduler_t* scheduler, float anomaly_score, bool* out_state_changed) {
    if (!scheduler) return false;

    scheduler->total_windows_processed++;
    bool state_changed = false;
    bool trigger_alert = false;

    if (anomaly_score >= scheduler->threshold) {
        /* Anomaly detected */
        scheduler->consecutive_normal_count = 0;
        if (scheduler->current_state != SYSTEM_STATE_ACTIVE) {
            scheduler->current_state = SYSTEM_STATE_ACTIVE;
            scheduler->state_transition_count++;
            state_changed = true;
        }
        scheduler->total_active_windows++;
        trigger_alert = true;
    } else {
        /* Normal beat observed */
        if (scheduler->current_state == SYSTEM_STATE_ACTIVE) {
            scheduler->consecutive_normal_count++;
            if (scheduler->consecutive_normal_count >= scheduler->recovery_count_target) {
                scheduler->current_state = SYSTEM_STATE_SLEEP;
                scheduler->state_transition_count++;
                state_changed = true;
                scheduler->consecutive_normal_count = 0;
            } else {
                scheduler->total_active_windows++;
            }
        }
        trigger_alert = false;
    }

    if (out_state_changed) {
        *out_state_changed = state_changed;
    }

    return trigger_alert;
}

const char* state_scheduler_get_state_str(SystemState_t state) {
    switch (state) {
        case SYSTEM_STATE_SLEEP:  return "SLEEP";
        case SYSTEM_STATE_ACTIVE: return "ACTIVE";
        default:                  return "UNKNOWN";
    }
}
