#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"
#include <math.h>

extern "C" {
JNIEXPORT float JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass);
}

#define F_S 48000
#define FRAME_SIZE 1024
#define VOICED_THRESHOLD 1e8
#define MIN_PITCH_FREQ 80
#define MAX_PITCH_FREQ 400

float lastFreqDetected = -1;

kiss_fft_cfg fft_cfg = NULL;
kiss_fft_cfg ifft_cfg = NULL;
kiss_fft_cpx fft_in[FRAME_SIZE * 2];
kiss_fft_cpx fft_out[FRAME_SIZE * 2];
kiss_fft_cpx ifft_out[FRAME_SIZE * 2];

void initFFT() {
    if (fft_cfg == NULL) {
        fft_cfg = kiss_fft_alloc(FRAME_SIZE * 2, 0, NULL, NULL);
        ifft_cfg = kiss_fft_alloc(FRAME_SIZE * 2, 1, NULL, NULL);
    }
}

void ece420ProcessFrame(sample_buf *dataBuf) {
    float bufferIn[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        int16_t val = ((uint16_t) dataBuf->buf_[2 * i]) |
                      (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);
        bufferIn[i] = (float) val;
    }

    float energy = 0.0f;
    for (int i = 0; i < FRAME_SIZE; i++) {
        energy += bufferIn[i] * bufferIn[i];
    }
    if (energy < VOICED_THRESHOLD) {
        lastFreqDetected = -1;
        return;
    }

    initFFT();
    for (int i = 0; i < FRAME_SIZE; i++) {
        fft_in[i].r = bufferIn[i];
        fft_in[i].i = 0.0f;
    }
    for (int i = FRAME_SIZE; i < FRAME_SIZE * 2; i++) {
        fft_in[i].r = 0.0f;
        fft_in[i].i = 0.0f;
    }

    kiss_fft(fft_cfg, fft_in, fft_out);
    for (int i = 0; i < FRAME_SIZE * 2; i++) {
        float real = fft_out[i].r;
        float imag = fft_out[i].i;
        fft_out[i].r = real * real + imag * imag;
        fft_out[i].i = 0.0f;
    }
    kiss_fft(ifft_cfg, fft_out, ifft_out);

    float norm = ifft_out[0].r;
    if (norm <= 0.0f) {
        lastFreqDetected = -1;
        return;
    }

    int minLag = (int)(F_S / MAX_PITCH_FREQ);
    int maxLag = (int)(F_S / MIN_PITCH_FREQ);
    if (maxLag >= FRAME_SIZE) maxLag = FRAME_SIZE - 1;
    if (minLag < 1) minLag = 1;

    int bestLag = -1;
    float threshold = 0.3f * norm;
    for (int lag = minLag; lag <= maxLag; lag++) {
        float val = ifft_out[lag].r;
        if (val > threshold && val > ifft_out[lag - 1].r && val > ifft_out[lag + 1].r) {
            bestLag = lag;
            break;
        }
    }

    if (bestLag > 0) {
        float freq = (float)F_S / (float)bestLag;
        if (freq >= MIN_PITCH_FREQ && freq <= MAX_PITCH_FREQ) {
            lastFreqDetected = freq;
            return;
        }
    }

    lastFreqDetected = -1;
}

JNIEXPORT float JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass) {
    return lastFreqDetected;
}
