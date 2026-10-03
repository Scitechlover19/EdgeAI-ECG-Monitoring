/**
 * @file ecg_dsp.c
 * @brief Embedded DSP filter implementation for Cortex-M4.
 */

#include "ecg_dsp.h"
#include <math.h>
#include <string.h>

void ecg_dsp_init(ECGDSPFilter_t* dsp) {
    if (!dsp) return;
    memset(dsp, 0, sizeof(ECGDSPFilter_t));

    /* 2nd-order Butterworth Highpass @ 0.5 Hz, Fs = 360 Hz */
    /* Pre-calculated direct-form II transposed biquad coefficients */
    dsp->hp1.b0 =  0.993863f;
    dsp->hp1.b1 = -1.987727f;
    dsp->hp1.b2 =  0.993863f;
    dsp->hp1.a1 = -1.987707f;
    dsp->hp1.a2 =  0.987747f;
    dsp->hp1.x1 = 0.0f; dsp->hp1.x2 = 0.0f;
    dsp->hp1.y1 = 0.0f; dsp->hp1.y2 = 0.0f;

    /* 2nd-order Butterworth Lowpass @ 45.0 Hz, Fs = 360 Hz */
    dsp->lp1.b0 =  0.115926f;
    dsp->lp1.b1 =  0.231853f;
    dsp->lp1.b2 =  0.115926f;
    dsp->lp1.a1 = -0.923880f;
    dsp->lp1.a2 =  0.387586f;
    dsp->lp1.x1 = 0.0f; dsp->lp1.x2 = 0.0f;
    dsp->lp1.y1 = 0.0f; dsp->lp1.y2 = 0.0f;

    dsp->initialized = true;
}

void ecg_dsp_reset(ECGDSPFilter_t* dsp) {
    if (!dsp) return;
    dsp->hp1.x1 = 0.0f; dsp->hp1.x2 = 0.0f;
    dsp->hp1.y1 = 0.0f; dsp->hp1.y2 = 0.0f;
    dsp->lp1.x1 = 0.0f; dsp->lp1.x2 = 0.0f;
    dsp->lp1.y1 = 0.0f; dsp->lp1.y2 = 0.0f;
}

static inline float biquad_step(BiquadSection_t* s, float in) {
    float out = s->b0 * in + s->b1 * s->x1 + s->b2 * s->x2 - s->a1 * s->y1 - s->a2 * s->y2;
    s->x2 = s->x1;
    s->x1 = in;
    s->y2 = s->y1;
    s->y1 = out;
    return out;
}

float ecg_dsp_process_sample(ECGDSPFilter_t* dsp, float raw_sample) {
    if (!dsp || !dsp->initialized) return raw_sample;
    float hp_out = biquad_step(&dsp->hp1, raw_sample);
    float bp_out = biquad_step(&dsp->lp1, hp_out);
    return bp_out;
}

void ecg_dsp_process_window(ECGDSPFilter_t* dsp, const float* input_raw, float* output_norm, uint16_t length) {
    if (!dsp || !input_raw || !output_norm || length == 0) return;

    /* Step 1: Bandpass filter the raw samples */
    float sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        output_norm[i] = ecg_dsp_process_sample(dsp, input_raw[i]);
        sum += output_norm[i];
    }

    /* Step 2: Z-score normalization */
    float mean = sum / (float)length;
    float sq_diff_sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float diff = output_norm[i] - mean;
        sq_diff_sum += diff * diff;
    }
    float std_dev = sqrtf(sq_diff_sum / (float)length);
    if (std_dev < 1e-6f) {
        std_dev = 1.0f;
    }

    for (uint16_t i = 0; i < length; i++) {
        output_norm[i] = (output_norm[i] - mean) / std_dev;
    }
}
