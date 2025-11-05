"""
Configuration parameters for physio processing.
Contains MATLAB-compatible parameters for Eulerian Video Magnification.
"""

# Frequency bands (Hz) based on paper specifications
PULSE_BAND_HZ = (0.5, 3.0)      # ~30-180 BPM (full physiological range)
PULSE_BAND_WIDE_HZ = (0.4, 4.0)  # Extended range for validation
RESP_BAND_HZ = (0.2, 0.5)       # ~12-30 breaths/min

# MATLAB-compatible video processing configurations
# Based on reproduceResults.m from the original MATLAB implementation
VIDEO_CONFIGS = {
    # Videos that use Gaussian pyramid + Ideal temporal filtering (Python compatible)
    'baby2': {
        'method': 'gaussian_ideal',
        'alpha': 150,
        'levels': 6,
        'fl': 140/60,  # 140 BPM to Hz
        'fh': 160/60,  # 160 BPM to Hz
        'chrom_attenuation': 1.0,
        'description': 'Baby video with heart rate amplification (140-160 BPM)'
    },
    'face': {
        'method': 'gaussian_ideal', 
        'alpha': 50,
        'levels': 4,
        'fl': 50/60,   # 50 BPM to Hz
        'fh': 60/60,   # 60 BPM to Hz
        'chrom_attenuation': 1.0,
        'description': 'Face video with pulse amplification (50-60 BPM)'
    },
    
    # Videos that use Laplacian pyramid in MATLAB (not yet implemented in Python)
    # These would need amplify_spatial_lpyr_temporal_butter/ideal/iir methods
    'baby': {
        'method': 'laplacian_iir',  # Not implemented in Python yet
        'alpha': 10,
        'levels': 16,
        'fl': 0.4,
        'fh': 3.0,
        'lambda_c': 30,
        'description': 'Baby video with IIR temporal filtering (Laplacian pyramid)'
    },
    'camera': {
        'method': 'laplacian_butter',  # Not implemented in Python yet
        'alpha': 150,
        'levels': 20,
        'fl': 45,
        'fh': 100,
        'lambda_c': 300,
        'description': 'Camera video with Butterworth temporal filtering (motion amplification)'
    },
    'subway': {
        'method': 'laplacian_butter',  # Not implemented in Python yet
        'alpha': 60,
        'levels': 90,
        'fl': 3.6,
        'fh': 6.2,
        'lambda_c': 30,
        'description': 'Subway video with Butterworth temporal filtering'
    },
    'wrist': {
        'method': 'laplacian_iir',  # Not implemented in Python yet
        'alpha': 10,
        'levels': 16,
        'fl': 0.4,
        'fh': 3.0,
        'lambda_c': 30,
        'description': 'Wrist video with IIR temporal filtering'
    },
    'shadow': {
        'method': 'laplacian_butter',  # Not implemented in Python yet
        'alpha': 5,
        'levels': 48,
        'fl': 0.5,
        'fh': 10,
        'lambda_c': 30,
        'description': 'Shadow video with Butterworth temporal filtering'
    },
    'guitar': {
        'method': 'laplacian_ideal',  # Not implemented in Python yet
        'alpha': 50,  # For E string (72-92 Hz)
        'levels': 10,
        'fl': 72,
        'fh': 92,
        'lambda_c': 600,
        'description': 'Guitar video with Ideal temporal filtering (string vibration)'
    },
    'face2': {
        'method': 'laplacian_butter',  # Not implemented in Python yet (motion)
        'alpha': 20,
        'levels': 80,
        'fl': 0.5,
        'fh': 10,
        'lambda_c': 30,
        'description': 'Face2 video with Butterworth temporal filtering (motion)'
    }
}

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