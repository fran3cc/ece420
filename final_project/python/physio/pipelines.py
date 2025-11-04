"""
Core processing pipelines for pulse and respiration detection.
"""

import numpy as np
from .types import FrameBatch, SignalSeries, Spectrum, RateEstimate, Diagnostics
from .roi import ROISelector
from .filters import moving_average_detrend, butter_bandpass
from .spectrum import fft_spectrum, pick_peak, hz_to_bpm, hz_to_breaths
from .config import PULSE_BAND_HZ, RESP_BAND_HZ, DETREND_WINDOW_SEC, DETREND_WINDOW_RESP_SEC


def extract_roi_signal(frames_bgr: list[np.ndarray], roi_selector: ROISelector, 
                      channel: str = 'green') -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract temporal signal from ROI across frames.
    
    Args:
        frames_bgr: List of BGR frames
        roi_selector: ROI selection strategy
        channel: 'green', 'luma', or 'red'
        
    Returns:
        Tuple of (signal_values, roi_coordinates)
    """
    if not frames_bgr:
        return np.array([]), (0, 0, 0, 0)
    
    # Get ROI from first frame
    x, y, w, h = roi_selector.select(frames_bgr[0])
    
    # Extract signal from each frame
    values = []
    for frame in frames_bgr:
        # Ensure ROI is within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x_safe = max(0, min(x, frame_w - 1))
        y_safe = max(0, min(y, frame_h - 1))
        w_safe = max(1, min(w, frame_w - x_safe))
        h_safe = max(1, min(h, frame_h - y_safe))
        
        roi = frame[y_safe:y_safe+h_safe, x_safe:x_safe+w_safe]
        
        if channel == 'green':
            val = roi[:, :, 1].mean()  # Green channel (BGR format)
        elif channel == 'red':
            val = roi[:, :, 2].mean()  # Red channel (BGR format)
        elif channel == 'luma':
            # ITU-R BT.601 luma coefficients for BGR
            val = (0.114 * roi[:, :, 0].mean() + 
                   0.587 * roi[:, :, 1].mean() + 
                   0.299 * roi[:, :, 2].mean())
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        values.append(float(val))
    
    return np.array(values), (x, y, w, h)


def run_pulse(batch: FrameBatch, roi: ROISelector, 
              band: tuple[float, float] = None) -> tuple[RateEstimate, Spectrum, SignalSeries, Diagnostics]:
    """
    Run pulse detection pipeline.
    
    Args:
        batch: Input video frames with FPS
        roi: ROI selector for face/forehead region
        band: Frequency band (fmin, fmax) in Hz
        
    Returns:
        Tuple of (rate_estimate, spectrum, signal, diagnostics)
    """
    if band is None:
        band = PULSE_BAND_HZ
    
    # Extract green channel signal from ROI
    signal_values, roi_coords = extract_roi_signal(batch.frames_bgr, roi, channel='green')
    signal = SignalSeries(values=signal_values, fps=batch.fps)
    
    # Create diagnostics
    diagnostics = Diagnostics(roi=roi_coords)
    
    if len(signal_values) < 4:
        # Not enough data
        return (RateEstimate(rate=0.0, freq_hz=0.0), 
                Spectrum(freqs_hz=np.array([0.0]), magnitude=np.array([0.0])),
                signal, diagnostics)
    
    # Signal processing pipeline
    # Step 1: Detrend
    detrended = moving_average_detrend(signal_values, batch.fps, DETREND_WINDOW_SEC)
    
    # Step 2: Band-pass filter
    filtered = butter_bandpass(detrended, batch.fps, band[0], band[1])
    
    # Step 3: Frequency analysis
    freqs, magnitude = fft_spectrum(filtered, batch.fps)
    spectrum = Spectrum(freqs_hz=freqs, magnitude=magnitude)
    
    # Step 4: Peak detection
    f_peak = pick_peak(freqs, magnitude, band[0], band[1])
    
    # Step 5: Convert to BPM
    bpm = hz_to_bpm(f_peak)
    
    return RateEstimate(rate=bpm, freq_hz=f_peak), spectrum, signal, diagnostics


def run_respiration(batch: FrameBatch, roi: ROISelector,
                   band: tuple[float, float] = None) -> tuple[RateEstimate, Spectrum, SignalSeries, Diagnostics]:
    """
    Run respiration detection pipeline.
    
    Args:
        batch: Input video frames with FPS
        roi: ROI selector for chest region
        band: Frequency band (fmin, fmax) in Hz
        
    Returns:
        Tuple of (rate_estimate, spectrum, signal, diagnostics)
    """
    if band is None:
        band = RESP_BAND_HZ
    
    # Extract luminance signal from ROI
    signal_values, roi_coords = extract_roi_signal(batch.frames_bgr, roi, channel='luma')
    signal = SignalSeries(values=signal_values, fps=batch.fps)
    
    # Create diagnostics
    diagnostics = Diagnostics(roi=roi_coords)
    
    if len(signal_values) < 4:
        # Not enough data
        return (RateEstimate(rate=0.0, freq_hz=0.0),
                Spectrum(freqs_hz=np.array([0.0]), magnitude=np.array([0.0])),
                signal, diagnostics)
    
    # Signal processing pipeline
    # Step 1: Detrend (longer window for respiration)
    detrended = moving_average_detrend(signal_values, batch.fps, DETREND_WINDOW_RESP_SEC)
    
    # Step 2: Band-pass filter
    filtered = butter_bandpass(detrended, batch.fps, band[0], band[1])
    
    # Step 3: Frequency analysis
    freqs, magnitude = fft_spectrum(filtered, batch.fps)
    spectrum = Spectrum(freqs_hz=freqs, magnitude=magnitude)
    
    # Step 4: Peak detection
    f_peak = pick_peak(freqs, magnitude, band[0], band[1])
    
    # Step 5: Convert to breaths/min
    breaths_per_min = hz_to_breaths(f_peak)
    
    return RateEstimate(rate=breaths_per_min, freq_hz=f_peak), spectrum, signal, diagnostics