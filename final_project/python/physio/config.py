"""
Configuration parameters for physio processing.
"""

# Frequency bands (Hz) based on paper specifications
PULSE_BAND_HZ = (0.5, 3.0)      # ~30-180 BPM (full physiological range)
PULSE_BAND_WIDE_HZ = (0.4, 4.0)  # Extended range for validation
RESP_BAND_HZ = (0.2, 0.5)       # ~12-30 breaths/min

# Signal processing parameters
DETREND_WINDOW_SEC = 4.0         # Moving average window for pulse
DETREND_WINDOW_RESP_SEC = 5.0    # Moving average window for respiration
FILTER_ORDER = 3                 # Butterworth filter order

# Spectrum analysis parameters
MIN_SIGNAL_LENGTH_SEC = 8.0      # Minimum signal length for reliable analysis
FFT_ZERO_PAD_FACTOR = 2          # Zero padding factor for FFT
PEAK_PROMINENCE_THRESHOLD = 0.1  # Minimum peak prominence (relative)

# Live camera parameters
LIVE_WINDOW_SEC = 15.0           # Rolling window size for live processing
LIVE_UPDATE_INTERVAL_SEC = 0.5   # How often to update rate estimate
EMA_SMOOTHING_FACTOR = 0.3       # Exponential moving average for rate smoothing

# ROI parameters
FACE_ROI_RELATIVE = (0.25, 0.17, 0.5, 0.2)    # (x, y, w, h) as fraction of frame
CHEST_ROI_RELATIVE = (0.25, 0.5, 0.5, 0.33)   # (x, y, w, h) as fraction of frame