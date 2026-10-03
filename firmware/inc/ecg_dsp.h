/**
 * @file ecg_dsp.h
 * @brief Embedded DSP preprocessing stage: Bandpass filtering & Z-score normalization.
 *
 * Implements deterministic 0.5 - 45 Hz bandpass filtering and amplitude scaling.
 * Governed by PRD.md (FR-3, FR-4) and Architecture.md §5.2.
 */

#ifndef ECG_DSP_H_
#define ECG_DSP_H_

#include <stdint.h>
#include <stdbool.h>
#include "mcu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
    float y1, y2;
} BiquadSection_t;

typedef struct {
    BiquadSection_t hp1; /* Highpass stage (~0.5 Hz) */
    BiquadSection_t lp1; /* Lowpass stage (~45 Hz) */
    bool initialized;
} ECGDSPFilter_t;

/**
 * @brief Initialize DSP filter coefficients for 360 Hz sampling.
 */
void ecg_dsp_init(ECGDSPFilter_t* dsp);

/**
 * @brief Reset internal filter delay lines.
 */
void ecg_dsp_reset(ECGDSPFilter_t* dsp);

/**
 * @brief Process single sample through bandpass filter.
 */
float ecg_dsp_process_sample(ECGDSPFilter_t* dsp, float raw_sample);

/**
 * @brief Process full window of 200 samples with bandpass and Z-score normalization.
 * 
 * @param dsp Pointer to filter instance
 * @param input_raw Array of 200 raw ECG samples (float)
 * @param output_norm Array of 200 filtered and normalized samples (float)
 */
void ecg_dsp_process_window(ECGDSPFilter_t* dsp, const float* input_raw, float* output_norm, uint16_t length);

#ifdef __cplusplus
}
#endif

#endif /* ECG_DSP_H_ */
