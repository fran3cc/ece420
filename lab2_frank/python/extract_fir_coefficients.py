import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Your exact filter design parameters
Fs = 48000
nyq = Fs / 2
numtaps = 201
desired = [1,1,0,0,1,1]
weights = [1,10,1]

bands = [0, 600, 1000, 2000, 2400, nyq]
b = signal.firls(numtaps, bands, desired, weights, fs=Fs)

print(f"Filter coefficients extracted:")
print(f"Number of taps (N_TAPS): {len(b)}")
print(f"Filter order: {len(b)-1}")

# Generate C++ code
print("\n=== C++ Code for ece420_main.cpp ===")
print(f"#define N_TAPS {len(b)}")
print("\nfloat myfilter[N_TAPS] = {")

# Format coefficients for C++ (8 per line)
for i in range(0, len(b), 8):
    line_coeffs = b[i:i+8]
    formatted_coeffs = [f"{coeff:.10f}f" for coeff in line_coeffs]
    if i + 8 >= len(b):  # Last line
        print("    " + ", ".join(formatted_coeffs))
    else:
        print("    " + ", ".join(formatted_coeffs) + ",")

print("};")

# Verify filter performance
w, h = signal.freqz(b, fs=Fs)

# Check attenuation at key frequencies
freq_1500 = np.argmin(np.abs(w - 1500))  # 1.5 kHz
attenuation_1500 = 20 * np.log10(abs(h[freq_1500]))

print(f"\n=== Filter Performance ===")
print(f"Attenuation at 1.5 kHz: {attenuation_1500:.2f} dB")
print(f"Filter length: {len(b)} taps")
print(f"Computational load: {len(b)} multiplications per sample")

# Plot frequency response
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.title(f'FIR Bandstop Filter Frequency Response (N = {len(b)})')
plt.plot(w, 20 * np.log10(abs(h)), 'b', linewidth=2)
plt.axhline(-20, color='r', linestyle='--', alpha=0.7, label='-20 dB spec')
plt.axvline(1000, color='g', linestyle='--', alpha=0.7, label='1 kHz')
plt.axvline(2000, color='g', linestyle='--', alpha=0.7, label='2 kHz')
plt.ylabel('Magnitude [dB]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 8000)
plt.ylim(-80, 5)

plt.subplot(2,1,2)
plt.plot(w, np.unwrap(np.angle(h)), 'g', linewidth=2)
plt.ylabel('Phase [radians]')
plt.xlabel('Frequency [Hz]')
plt.grid(True, alpha=0.3)
plt.xlim(0, 8000)

plt.tight_layout()
plt.show()

# Generate test chirp signal
print("\n=== Generating Test Signals ===")
duration = 3.0  # seconds
t = np.linspace(0, duration, int(Fs * duration), False)

# Chirp from 0 to 3000 Hz
chirp_signal = np.sin(2 * np.pi * np.linspace(0, 3000, len(t)) * t)

# Apply filter
filtered_chirp = signal.lfilter(b, 1, chirp_signal)

# Save audio files
from scipy.io import wavfile

# Normalize and convert to 16-bit
original_audio = np.int16(chirp_signal * 32767 * 0.8)
filtered_audio = np.int16(filtered_chirp * 32767 * 0.8)

wavfile.write('original_chirp.wav', Fs, original_audio)
wavfile.write('filtered_chirp.wav', Fs, filtered_audio)

print("Audio files saved:")
print("- original_chirp.wav (0-3000 Hz chirp)")
print("- filtered_chirp.wav (filtered version)")

# Spectral analysis
from scipy.signal import spectrogram

f_orig, t_orig, Sxx_orig = spectrogram(chirp_signal, Fs, nperseg=1024)
f_filt, t_filt, Sxx_filt = spectrogram(filtered_chirp, Fs, nperseg=1024)

plt.figure(figsize=(15, 10))

plt.subplot(2,2,1)
plt.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig), shading='gouraud')
plt.title('Original Chirp Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 3500)
plt.colorbar(label='Power [dB]')

plt.subplot(2,2,2)
plt.pcolormesh(t_filt, f_filt, 10 * np.log10(Sxx_filt), shading='gouraud')
plt.title('Filtered Chirp Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 3500)
plt.colorbar(label='Power [dB]')

plt.subplot(2,2,3)
plt.plot(t[:Fs], chirp_signal[:Fs], 'b', alpha=0.7, label='Original')
plt.plot(t[:Fs], filtered_chirp[:Fs], 'r', alpha=0.7, label='Filtered')
plt.title('Time Domain Comparison (First 1 second)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,4)
# FFT analysis of middle section (1-2 kHz region)
mid_start = int(len(t) * 0.33)  # Around 1 kHz
mid_end = int(len(t) * 0.67)    # Around 2 kHz

fft_orig = np.fft.fft(chirp_signal[mid_start:mid_end])
fft_filt = np.fft.fft(filtered_chirp[mid_start:mid_end])
freqs = np.fft.fftfreq(len(fft_orig), 1/Fs)

positive_freqs = freqs[:len(freqs)//2]
plt.plot(positive_freqs, 20*np.log10(np.abs(fft_orig[:len(freqs)//2])), 'b', alpha=0.7, label='Original')
plt.plot(positive_freqs, 20*np.log10(np.abs(fft_filt[:len(freqs)//2])), 'r', alpha=0.7, label='Filtered')
plt.title('FFT Analysis (1-2 kHz Region)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.xlim(500, 2500)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== Summary ===")
print(f"Your 201-tap FIR filter provides {attenuation_1500:.1f} dB attenuation at 1.5 kHz")
print(f"Computational cost: {len(b)} multiplications per sample")
print(f"For 48 kHz sampling: {len(b) * 48000:,} operations per second")
print(f"\nThe generated C++ code can be directly used in your ece420_main.cpp file.")