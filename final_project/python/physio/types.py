"""
Core data types for the physio package.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class FrameBatch:
    """A batch of video frames with associated frame rate."""
    frames_bgr: list[np.ndarray]
    fps: float


@dataclass
class SignalSeries:
    """A temporal signal series with associated sampling rate."""
    values: np.ndarray
    fps: float


@dataclass
class Spectrum:
    """Frequency spectrum data."""
    freqs_hz: np.ndarray
    magnitude: np.ndarray


@dataclass
class RateEstimate:
    """Estimated physiological rate (BPM or breaths/min)."""
    rate: float       # BPM or breaths/min
    freq_hz: float    # Peak frequency in Hz


@dataclass
class Diagnostics:
    """Diagnostic information from processing."""
    roi: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)
    notes: Optional[Dict[str, Any]] = None