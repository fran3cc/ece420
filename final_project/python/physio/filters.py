"""Signal processing filters for physiological signals."""

import numpy as np

# Try to import scipy, fall back to simple implementations if not available
try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available, using simple filter implementations")


def moving_average_detrend(x: np.ndarray, fps: float, window_sec: float = 4.0) -> np.ndarray:
    """
    Remove slow trends using moving average subtraction.
    
    Args:
        x: Input signal
        fps: Sampling rate in Hz
        window_sec: Window size in seconds for moving average
        
    Returns:
        Detrended signal
    """
    if len(x) < 2:
        return x.copy()
    
    # Calculate window size in samples
    window_samples = max(1, int(window_sec * fps))
    
    # Create moving average kernel
    kernel = np.ones(window_samples) / window_samples
    
    # Apply convolution to get trend
    trend = np.convolve(x, kernel, mode='same')
    
    # Handle edge effects by using simpler averaging at boundaries
    half_window = window_samples // 2
    if half_window > 0:
        # Left edge
        for i in range(half_window):
            trend[i] = np.mean(x[:i+half_window+1])
        # Right edge
        for i in range(len(x) - half_window, len(x)):
            trend[i] = np.mean(x[i-half_window:])
    
    return x - trend


def butter_bandpass(x: np.ndarray, fps: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to the signal.
    
    Args:
        x: Input signal
        fps: Sampling rate in Hz
        low_hz: Low cutoff frequency in Hz
        high_hz: High cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if len(x) < 2 * order:
        return x  # Signal too short for filtering
    
    if SCIPY_AVAILABLE:
        nyquist = fps / 2
        low = low_hz / nyquist
        high = high_hz / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, x)
    else:
        # Simple fallback: high-pass then low-pass using moving averages
        # This is not as good as a proper Butterworth filter but will work
        
        # High-pass: subtract low-frequency component
        low_window = int(fps / low_hz) if low_hz > 0 else len(x) // 4
        low_window = max(3, min(low_window, len(x) // 3))
        
        # Simple moving average for low-frequency component
        kernel = np.ones(low_window) / low_window
        if len(x) >= low_window:
            low_freq = np.convolve(x, kernel, mode='same')
            high_passed = x - low_freq
        else:
            high_passed = x - np.mean(x)
        
        # Low-pass: smooth high-frequency noise
        high_window = int(fps / high_hz) if high_hz > 0 else 3
        high_window = max(3, min(high_window, len(x) // 3))
        
        if len(high_passed) >= high_window:
            kernel = np.ones(high_window) / high_window
            filtered = np.convolve(high_passed, kernel, mode='same')
        else:
            filtered = high_passed
            
        return filtered