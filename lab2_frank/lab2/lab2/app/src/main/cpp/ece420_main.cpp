//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include "ece420_main.h"

// Student Variables
#define FRAME_SIZE 128

// FIR Filter Function Defined here located at the bottom
int16_t firFilter(int16_t sample);

void ece420ProcessFrame(sample_buf *dataBuf) {
    // Keep in mind, we only have a small amount of time to process each buffer!
    struct timeval start;
    gettimeofday(&start, NULL);

    // Using {} initializes all values in the array to zero
    int16_t bufferIn[FRAME_SIZE] = {};
    int16_t bufferOut[FRAME_SIZE] = {};

    // Your buffer conversion (unpacking) here
    // Fetch data sample from dataBuf->buf_[], unpack and put into bufferIn[]
    // ******************** START YOUR CODE HERE ******************** //
    for (int i = 0; i < FRAME_SIZE; i++) {
        // PCM-16 data is stored as 2 bytes per sample (little-endian)
        int16_t sample = (int16_t)((dataBuf->buf_[2*i+1] << 8) | dataBuf->buf_[2*i]);
        bufferIn[i] = sample;
    }

    // ********************* END YOUR CODE HERE ********************* //

    // Loop code provided as a suggestion. This loop simulates sample-by-sample processing.
    for (int sampleIdx = 0; sampleIdx < FRAME_SIZE; sampleIdx++) {
        // Grab one sample from bufferIn[]
        int16_t sample = bufferIn[sampleIdx];
        // Call your filFilter funcion
        int16_t output = firFilter(sample);
        // Grab result and put into bufferOut[]
        bufferOut[sampleIdx] = output;
    }

    // Your buffer conversion (packing) here
    // Fetch data sample from bufferOut[], pack them and put back into dataBuf->buf_[]
    // ******************** START YOUR CODE HERE ******************** //
    for (int i = 0; i < FRAME_SIZE; i++) {
        // Convert int16_t back to PCM-16 (little-endian, 2 bytes per sample)
        dataBuf->buf_[2*i] = (uint8_t)(bufferOut[i] & 0xFF);        // Low byte
        dataBuf->buf_[2*i+1] = (uint8_t)((bufferOut[i] >> 8) & 0xFF); // High byte
    }


    // ********************* END YOUR CODE HERE ********************* //

	// Log the processing time to Android Monitor or Logcat window at the bottom
    struct timeval end;
    gettimeofday(&end, NULL);
    LOGD("Loop timer: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

}

// TODO: Change N_TAPS to match your filter design
#define N_TAPS 5
// TODO: Change myfilter to contain the coefficients of your designed filter.
double myfilter[N_TAPS] = {1,0,0,0,0};

// Circular Buffer
int16_t circBuf[N_TAPS] = {};
int16_t circBufIdx = 0;

// FirFilter Function
int16_t firFilter(int16_t sample) {
    // This function simulates sample-by-sample processing. Here you will
    // implement an FIR filter such as:
    //
    // y[n] = a x[n] + b x[n-1] + c x[n-2] + ...
    //
    // You will maintain a circular buffer to store your prior samples
    // x[n-1], x[n-2], ..., x[n-k]. Suggested initializations circBuf
    // and circBufIdx are given.
    //
    // Input 'sample' is the current sample x[n].
    // ******************** START YOUR CODE HERE ******************** //
    int16_t output = 0;
        // PART 5 - FIR FILTER IMPLEMENTATION
    // Store current sample in circular buffer
    circBuf[circBufIdx] = sample;
    
    // Apply FIR filter: y[n] = sum(h[k] * x[n-k])
    for (int i = 0; i < N_TAPS; i++) {
        // Calculate circular buffer index for x[n-i]
        int bufIdx = (circBufIdx - i + N_TAPS) % N_TAPS;
        // Accumulate filter output (be careful of overflow)
        output += (int16_t)(myfilter[i] * circBuf[bufIdx]);
    }
    
    // Update circular buffer index for next sample
    circBufIdx = (circBufIdx + 1) % N_TAPS;
    
    // PART 6 - IIR FILTER (EXTRA CREDIT)
    // Uncomment and modify the following section for IIR implementation:
    /*
    // Additional static variables needed for IIR (add these at top of file):
    // #define IIR_ORDER 2
    // static double b_coeffs[IIR_ORDER+1] = {0.25, 0.5, 0.25};  // Feedforward coefficients
    // static double a_coeffs[IIR_ORDER+1] = {1.0, -0.5, 0.25};  // Feedback coefficients (a[0] = 1.0)
    // static int16_t x_history[IIR_ORDER+1] = {0}; // Input history
    // static int16_t y_history[IIR_ORDER+1] = {0}; // Output history
    // static int iir_idx = 0;
    
    // Store current input
    x_history[iir_idx] = sample;
    
    // Calculate IIR output
    double iir_output = 0.0;
    
    // Feedforward part: sum(b[k] * x[n-k])
    for(int i = 0; i <= IIR_ORDER; i++) {
        int idx = (iir_idx - i + IIR_ORDER + 1) % (IIR_ORDER + 1);
        iir_output += b_coeffs[i] * x_history[idx];
    }
    
    // Feedback part: sum(a[k] * y[n-k]) for k > 0
    for(int i = 1; i <= IIR_ORDER; i++) {
        int idx = (iir_idx - i + IIR_ORDER + 1) % (IIR_ORDER + 1);
        iir_output -= a_coeffs[i] * y_history[idx];
    }
    
    // Store output and update index
    output = (int16_t)iir_output;
    y_history[iir_idx] = output;
    iir_idx = (iir_idx + 1) % (IIR_ORDER + 1);
    */


    // ********************* END YOUR CODE HERE ********************* //
    return output;
}
