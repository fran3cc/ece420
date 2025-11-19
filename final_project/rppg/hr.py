import numpy as np

def estimate_hr_bpm(x, fps):
    n = len(x)
    if n < 8:
        return 0.0
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, d=1.0/float(fps))
    mask = (freqs >= 0.8) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0
    mag = np.abs(X)
    idx = np.argmax(mag[mask])
    base_freq = freqs[mask][idx]
    base_bpm = base_freq * 60.0
    double_freq = base_freq * 2.0
    if base_bpm < 60.0 and double_freq <= freqs[-1]:
        di = np.argmin(np.abs(freqs - double_freq))
        if mag[di] > mag[idx]:
            base_freq = double_freq
    return float(base_freq*60.0)