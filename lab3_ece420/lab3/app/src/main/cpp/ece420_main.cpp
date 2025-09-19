//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include <jni.h>
#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"

// Declare JNI function
extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab3_MainActivity_getFftBuffer(JNIEnv *env, jclass, jobject bufferPtr);
}

// FRAME_SIZE is 1024 and we zero-pad it to 2048 to do FFT
#define FRAME_SIZE 1024
#define ZP_FACTOR 2
#define FFT_SIZE (FRAME_SIZE * ZP_FACTOR)

// Variable to store final FFT output
float fftOut[FFT_SIZE] = {};
bool isWritingFft = false;

void ece420ProcessFrame(sample_buf *dataBuf) {
    // Block UI thread from reading
    isWritingFft = true;

    // We only have ~20ms to process each buffer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Convert PCM-16 input buffer to float
    float bufferIn[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        int16_t val = ((uint16_t)dataBuf->buf_[2 * i]) |
                      (((uint16_t)dataBuf->buf_[2 * i + 1]) << 8);
        bufferIn[i] = (float)val;
    }

    // ================= FFT pipeline =================
    kiss_fft_cpx in[FFT_SIZE];
    kiss_fft_cpx out[FFT_SIZE];

    // 1. Apply Hamming window
    for (int i = 0; i < FRAME_SIZE; i++) {
        float w = 0.54f - 0.46f * cosf((2.0f * M_PI * i) / (FRAME_SIZE - 1));
        in[i].r = bufferIn[i] * w;
        in[i].i = 0.0f;
    }

    // 2. Zero-pad
    for (int i = FRAME_SIZE; i < FFT_SIZE; i++) {
        in[i].r = 0.0f;
        in[i].i = 0.0f;
    }

    // 3. Run FFT
    kiss_fft_cfg cfg = kiss_fft_alloc(FFT_SIZE, 0, NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);

    // 4. Magnitude squared
    float mag[FFT_SIZE / 2];
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        mag[i] = out[i].r * out[i].r + out[i].i * out[i].i;
    }

    // 5. Log scaling
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        mag[i] = logf(1.0f + mag[i]);
    }

    // 6. Normalize to [0,1]
    float maxVal = 0.0f;
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        if (mag[i] > maxVal) maxVal = mag[i];
    }

    if (maxVal > 0) {
        for (int i = 0; i < FRAME_SIZE; i++) {
            fftOut[i] = mag[i] / maxVal;
        }
    } else {
        for (int i = 0; i < FRAME_SIZE; i++) {
            fftOut[i] = 0.0f;
        }
    }
    // =================================================

    // Safe for UI thread to read
    isWritingFft = false;

    // Timing log
    gettimeofday(&end, NULL);
    LOGD("Time delay: %ld us",
         ((end.tv_sec * 1000000 + end.tv_usec) -
          (start.tv_sec * 1000000 + start.tv_usec)));
}

// JNI function to fetch FFT buffer for UI
JNIEXPORT void JNICALL
Java_com_ece420_lab3_MainActivity_getFftBuffer(JNIEnv *env, jclass, jobject bufferPtr) {
    jfloat *buffer = (jfloat *)env->GetDirectBufferAddress(bufferPtr);
    // Wait until not writing
    while (isWritingFft) {}
    // Copy FFT output (first FRAME_SIZE bins)
    for (int i = 0; i < FRAME_SIZE; i++) {
        buffer[i] = fftOut[i];
    }
}
