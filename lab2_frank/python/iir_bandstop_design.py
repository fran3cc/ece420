import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Design IIR bandstop filter
fs = 48000
nyquist = fs / 2

# Bandstop specifications
stop_low = 1000   # 1kHz
stop_high = 2000  # 2kHz

# Design 4th-order Butterworth bandstop
sos = signal.iirfilter(4, [stop_low, stop_high], btype='bandstop', 
                      ftype='butter', fs=fs, output='sos')

# Convert to transfer function for coefficient extraction
b, a = signal.sos2tf(sos)

print(f"=== IIR BANDSTOP FILTER DESIGN ===")
print(f"Filter order: {len(a)-1}")
print(f"Numerator coefficients (b): {len(b)}")
print(f"Denominator coefficients (a): {len(a)}")

# Analyze frequency response
w, h = signal.freqz(b, a, fs=fs)

# Check performance
freq_1500 = np.argmin(np.abs(w - 1500))
freq_1000 = np.argmin(np.abs(w - 1000))
freq_2000 = np.argmin(np.abs(w - 2000))
freq_500 = np.argmin(np.abs(w - 500))
freq_3000 = np.argmin(np.abs(w - 3000))

attenuation_1500 = 20 * np.log10(np.abs(h[freq_1500]))
attenuation_1000 = 20 * np.log10(np.abs(h[freq_1000]))
attenuation_2000 = 20 * np.log10(np.abs(h[freq_2000]))
gain_500 = 20 * np.log10(np.abs(h[freq_500]))
gain_3000 = 20 * np.log10(np.abs(h[freq_3000]))

print(f"\n=== PERFORMANCE ===")
print(f"Attenuation at 1.5kHz: {attenuation_1500:.1f} dB")
print(f"Attenuation at 1.0kHz: {attenuation_1000:.1f} dB")
print(f"Attenuation at 2.0kHz: {attenuation_2000:.1f} dB")
print(f"Gain at 500Hz: {gain_500:.1f} dB")
print(f"Gain at 3000Hz: {gain_3000:.1f} dB")

# Generate C++ code
print(f"\n=== C++ CODE FOR ece420_main.cpp ===")
print(f"// IIR Filter coefficients")
print(f"#define IIR_ORDER {len(a)-1}")
print(f"\n// Feedforward coefficients (numerator)")
print(f"double b_coeffs[{len(b)}] = {{")
for i, coeff in enumerate(b):
    if i == len(b)-1:
        print(f"    {coeff:.10f}")
    else:
        print(f"    {coeff:.10f},")
print("};")

print(f"\n// Feedback coefficients (denominator)")
print(f"double a_coeffs[{len(a)}] = {{")
for i, coeff in enumerate(a):
    if i == len(a)-1:
        print(f"    {coeff:.10f}")
    else:
        print(f"    {coeff:.10f},")
print("};")

# Test with chirp
duration = 3.0
t = np.linspace(0, duration, int(fs * duration), False)
chirp_signal = signal.chirp(t, f0=0, f1=3000, t1=duration, method='linear')

# Apply IIR filter
filtered_chirp = signal.lfilter(b, a, chirp_signal)

# Save audio files
wavfile.write('iir_original_chirp.wav', fs, (chirp_signal * 0.8).astype(np.float32))
wavfile.write('iir_filtered_chirp.wav', fs, (filtered_chirp * 0.8).astype(np.float32))

print("\n✓ Audio files saved: iir_original_chirp.wav, iir_filtered_chirp.wav")

# Plot comparison
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.axvline(1000, color='r', linestyle='--', alpha=0.7, label='1kHz')
plt.axvline(1500, color='r', linestyle='-', alpha=0.9, label='1.5kHz')
plt.axvline(2000, color='r', linestyle='--', alpha=0.7, label='2kHz')
plt.axhline(-20, color='g', linestyle=':', alpha=0.7, label='-20dB')
plt.title('IIR Bandstop Filter - Magnitude Response')
plt.ylabel('Magnitude (dB)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([-60, 5])

plt.subplot(2, 1, 2)
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.title('Phase Response')
plt.ylabel('Phase (radians)')
plt.xlabel('Frequency (Hz)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== COMPUTATIONAL COMPARISON ===")
print(f"FIR (121 taps): ~241 operations/sample")
print(f"IIR (4th order): ~{len(a) + len(b)} operations/sample")
print(f"Speed improvement: ~{241/(len(a) + len(b)):.0f}x faster")