"""Frequency domain analysis for physiological signals."""

import numpy as np

# Try to import scipy, fall back to simple implementations if not available
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def fft_spectrum(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT spectrum of signal.
    
    Args:
        x: Input signal
        fps: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies_hz, magnitude)
    """
    if len(x) < 2:
        return np.array([0.0]), np.array([0.0])
    
    # Compute FFT
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft_vals = np.fft.rfft(x)
    magnitude = np.abs(fft_vals)
    
    return freqs, magnitude


def welch_spectrum(x: np.ndarray, fps: float, nperseg: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch's method power spectral density.
    
    Args:
        x: Input signal
        fps: Sampling rate in Hz
        nperseg: Length of each segment for Welch's method
        
    Returns:
        Tuple of (frequencies_hz, power_spectral_density)
    """
    from scipy.signal import welch
    
    if len(x) < 4:
        return np.array([0.0]), np.array([0.0])
    
    if nperseg is None:
        nperseg = min(len(x) // 4, int(8 * fps))  # 8 second segments by default
    
    freqs, psd = welch(x, fs=fps, nperseg=nperseg)
    return freqs, psd


def pick_peak(freqs: np.ndarray, magnitude: np.ndarray, fmin: float, fmax: float, 
              prominence: float = 0.1) -> float:
    """
    Find the dominant peak within a frequency band.
    
    Args:
        freqs: Frequency array
        magnitude: Magnitude array
        fmin: Minimum frequency
        fmax: Maximum frequency
        prominence: Minimum peak prominence (relative to max)
        
    Returns:
        Peak frequency in Hz, or 0 if no peak found
    """
    # Find indices within the frequency band
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    
    band_freqs = freqs[mask]
    band_magnitude = magnitude[mask]
    
    if len(band_magnitude) < 3:
        # Not enough points for peak detection
        return band_freqs[np.argmax(band_magnitude)]
    
    if SCIPY_AVAILABLE:
        # Find peaks with minimum prominence
        min_prominence = prominence * np.max(band_magnitude)
        peaks, properties = find_peaks(band_magnitude, prominence=min_prominence)
        
        if len(peaks) == 0:
            # No prominent peaks found, return frequency of maximum
            return band_freqs[np.argmax(band_magnitude)]
        
        # Return the frequency of the most prominent peak
        most_prominent_idx = peaks[np.argmax(properties['prominences'])]
        return band_freqs[most_prominent_idx]
    else:
        # Simple fallback: find local maxima manually
        peaks = []
        min_prominence = prominence * np.max(band_magnitude)
        
        for i in range(1, len(band_magnitude) - 1):
            # Check if it's a local maximum
            if (band_magnitude[i] > band_magnitude[i-1] and 
                band_magnitude[i] > band_magnitude[i+1] and
                band_magnitude[i] > min_prominence):
                peaks.append(i)
        
        if len(peaks) == 0:
            # No prominent peaks found, return frequency of maximum
            return band_freqs[np.argmax(band_magnitude)]
        
        # Return the frequency of the highest peak
        peak_magnitudes = [band_magnitude[p] for p in peaks]
        highest_peak_idx = peaks[np.argmax(peak_magnitudes)]
        return band_freqs[highest_peak_idx]


def hz_to_bpm(f_hz: float) -> float:
    """Convert frequency in Hz to beats per minute."""
    return 60.0 * f_hz


def hz_to_breaths(f_hz: float) -> float:
    """Convert frequency in Hz to breaths per minute."""
    return 60.0 * f_hz