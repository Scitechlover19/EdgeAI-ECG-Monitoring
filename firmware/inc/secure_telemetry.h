/**
 * @file secure_telemetry.h
 * @brief Privacy-preserving metadata-only telemetry dispatch.
 *
 * CRITICAL RULE (AGENTS.md Rule 5):
 * NEVER TRANSMIT RAW ECG WAVEFORMS.
 * The payload is strictly restricted to discrete diagnostic metadata:
 * { timestamp_ms, anomaly_id, confidence_score, heart_rate_bpm, auth_tag }.
 *
 * Governed by PRD.md (FR-7, NFR-4) and Architecture.md §5.5 & §11.
 */

#ifndef SECURE_TELEMETRY_H_
#define SECURE_TELEMETRY_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TELEMETRY_MAX_JSON_LEN 256

/**
 * @brief Fixed metadata-only alert payload structure.
 * Statically guarantees NO raw ECG array pointer or buffer can be attached.
 */
typedef struct {
    uint32_t timestamp_ms;
    uint16_t anomaly_id;
    float confidence_score;
    float heart_rate_bpm;
    uint8_t auth_tag[16];   /* 128-bit AES-GCM tag or HMAC-SHA256 trunk */
} AlertMetadataPayload_t;

typedef struct {
    uint32_t total_alerts_transmitted;
    uint32_t total_bytes_transmitted;
} SecureTelemetry_t;

/**
 * @brief Initialize telemetry subsystem.
 */
void secure_telemetry_init(SecureTelemetry_t* telemetry);

/**
 * @brief Package and dispatch an encrypted metadata-only alert.
 * 
 * @param telemetry Pointer to telemetry instance
 * @param payload Metadata payload struct (validated, contains zero raw samples)
 * @param[out] out_json_buffer Buffer to receive serialized JSON packet
 * @param max_len Maximum length of out_json_buffer
 * @return int Number of bytes serialized, or negative on validation error
 */
int secure_telemetry_dispatch_alert(
    SecureTelemetry_t* telemetry,
    const AlertMetadataPayload_t* payload,
    char* out_json_buffer,
    uint16_t max_len
);

#ifdef __cplusplus
}
#endif

#endif /* SECURE_TELEMETRY_H_ */
