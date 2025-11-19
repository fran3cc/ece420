import numpy as np

def _fallback_bandpass(x, fps, low, high):
    n = len(x)
    x = x - np.mean(x)
    w = int(max(3, fps*1.5))
    k = min(w, n)
    ma = np.convolve(x, np.ones(k)/k, mode='same')
    hp = x - ma
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    X = np.fft.rfft(hp)
    mask = (freqs >= low) & (freqs <= high)
    X[~mask] = 0
    y = np.fft.irfft(X, n=n)
    return y

def detrend_and_bandpass(x, fps, low=0.8, high=3.0):
    try:
        from scipy.signal import butter, filtfilt, detrend
        x = detrend(x, type='linear')
        b, a = butter(3, [low/(fps/2.0), high/(fps/2.0)], btype='band')
        y = filtfilt(b, a, x)
        return y
    except Exception:
        return _fallback_bandpass(np.asarray(x, dtype=np.float64), float(fps), float(low), float(high))