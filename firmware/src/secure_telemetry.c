/**
 * @file secure_telemetry.c
 * @brief Secure metadata-only telemetry serialization.
 */

#include "secure_telemetry.h"
#include <stdio.h>
#include <string.h>

void secure_telemetry_init(SecureTelemetry_t* telemetry) {
    if (!telemetry) return;
    telemetry->total_alerts_transmitted = 0;
    telemetry->total_bytes_transmitted = 0;
}

int secure_telemetry_dispatch_alert(
    SecureTelemetry_t* telemetry,
    const AlertMetadataPayload_t* payload,
    char* out_json_buffer,
    uint16_t max_len
) {
    if (!payload || !out_json_buffer || max_len < 64) {
        return -1;
    }

    /* Format 16-byte hex authentication tag */
    char tag_hex[33] = {0};
    for (int i = 0; i < 16; i++) {
        snprintf(&tag_hex[i * 2], 3, "%02X", payload->auth_tag[i]);
    }

    /* Serialize strictly to metadata JSON. NO RAW ECG DATA IS EVER SERIALIZED. */
    int bytes = snprintf(
        out_json_buffer,
        max_len,
        "{\"msg\":\"ALERT\",\"ts\":%lu,\"anomaly_id\":%u,\"conf\":%.3f,\"bpm\":%.1f,\"auth\":\"%s\"}",
        (unsigned long)payload->timestamp_ms,
        (unsigned int)payload->anomaly_id,
        (double)payload->confidence_score,
        (double)payload->heart_rate_bpm,
        tag_hex
    );

    if (bytes > 0 && bytes < max_len) {
        if (telemetry) {
            telemetry->total_alerts_transmitted++;
            telemetry->total_bytes_transmitted += bytes;
        }
        return bytes;
    }

    return -1;
}
