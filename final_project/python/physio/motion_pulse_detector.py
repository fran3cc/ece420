"""
Motion-based pulse detection using optical flow and PCA (CVPR 2013 style).
Tracks vertical head motion via Lucas-Kanade optical flow, extracts periodic
component with PCA, and estimates BPM via FFT peak within pulse band.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple, List

from .roi import FaceROI, ROISelector
from .filters import moving_average_detrend, butter_bandpass
from .spectrum import fft_spectrum, pick_peak, hz_to_bpm


def track_head_motion_optical_flow(frames_bgr: List[np.ndarray], fps: float,
                                   roi_selector: ROISelector | None = None,
                                   max_corners: int = 60) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Track vertical motion (y-coordinates) of feature points within a face/forehead ROI.

    Returns a matrix of shape (n_frames, n_points) containing y positions over time,
    and the ROI coordinates (x, y, w, h).
    """
    if roi_selector is None:
        roi_selector = FaceROI()

    n_frames = len(frames_bgr)
    if n_frames == 0:
        return np.zeros((0, 0), dtype=np.float32), (0, 0, 0, 0)

    # Initialize
    first_bgr = frames_bgr[0]
    h, w = first_bgr.shape[:2]
    x, y, rw, rh = roi_selector.select(first_bgr)
    x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x)); rh = max(1, min(rh, h - y))

    # Prepare first frame and detect features inside ROI
    prev_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(prev_gray)
    mask[y:y+rh, x:x+rw] = 255
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                      qualityLevel=0.01, minDistance=5,
                                      mask=mask)
    if corners is None or len(corners) == 0:
        # fallback: grid points inside ROI
        xs = np.linspace(x + 4, x + rw - 4, num=8)
        ys = np.linspace(y + 4, y + rh - 4, num=8)
        grid = np.array([[xi, yi] for yi in ys for xi in xs], dtype=np.float32)
        corners = grid.reshape(-1, 1, 2)

    points_prev = corners.astype(np.float32)
    n_points = points_prev.shape[0]

    # Allocate y-traces
    traces_y = np.zeros((n_frames, n_points), dtype=np.float32)
    traces_y[0] = points_prev[:, 0, 1]

    # Track across frames
    for t in range(1, n_frames):
        frame_gray = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2GRAY)
        points_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, points_prev, None,
                                                        winSize=(21, 21), maxLevel=3,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if points_next is None:
            traces_y[t] = traces_y[t-1]
        else:
            st = st.reshape(-1)
            for i in range(n_points):
                if st[i]:
                    traces_y[t, i] = points_next[i, 0, 1]
                else:
                    traces_y[t, i] = traces_y[t-1, i]
            points_prev = points_next
        prev_gray = frame_gray

    return traces_y, (x, y, rw, rh)


def extract_pulse_from_motion(traces_y: np.ndarray, fps: float,
                              band: Tuple[float, float] = (0.8, 2.0),
                              detrend_win_sec: float = 3.0) -> Tuple[np.ndarray, float]:
    """
    Apply PCA to motion traces and select the most periodic component within the band.

    Returns (selected_component_time_series, bpm_estimate).
    """
    if traces_y.size == 0:
        return np.array([]), 0.0

    # Detrend each trace (remove slow drift), then band-pass in the pulse band
    detrended = np.zeros_like(traces_y, dtype=np.float32)
    filtered = np.zeros_like(traces_y, dtype=np.float32)
    for i in range(traces_y.shape[1]):
        detrended[:, i] = moving_average_detrend(traces_y[:, i], fps, detrend_win_sec)
        filtered[:, i] = butter_bandpass(detrended[:, i], fps, band[0], band[1])

    # Center columns and perform PCA via SVD on time x points matrix
    X = filtered - filtered.mean(axis=0, keepdims=True)
    # SVD: X = U S V^T, principal components time series = U S
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = U * S  # shape (n_frames, n_components)

    # Evaluate periodicity via max peak magnitude in band
    best_idx = 0
    best_peak_mag = -np.inf
    best_bpm = 0.0
    for k in range(components.shape[1]):
        comp = components[:, k]
        # Spectrum and peak
        freqs, mag = fft_spectrum(comp.astype(np.float64), fps)
        # Limit to band
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(band_mask):
            continue
        f_peak = pick_peak(freqs, mag, band[0], band[1])
        # Peak magnitude at f_peak
        idx = (np.abs(freqs - f_peak)).argmin()
        peak_mag = mag[idx]
        if peak_mag > best_peak_mag:
            best_peak_mag = peak_mag
            best_idx = k
            best_bpm = float(hz_to_bpm(f_peak))

    selected = components[:, best_idx] if components.shape[1] > 0 else np.array([])
    return selected.astype(np.float32), best_bpm


def calculate_bpm_from_motion(signal: np.ndarray, fps: float,
                               band: Tuple[float, float] = (0.8, 2.0)) -> float:
    """Estimate BPM from motion-derived signal via FFT peak in the band."""
    if signal.size == 0:
        return 0.0
    freqs, mag = fft_spectrum(signal.astype(np.float64), fps)
    f_peak = pick_peak(freqs, mag, band[0], band[1])
    return float(hz_to_bpm(f_peak))


def create_motion_pulse_overlay(signal: np.ndarray, fps: float, bpm: float,
                                 size: Tuple[int, int] = (320, 160)) -> List[np.ndarray]:
    """
    Create scrolling waveform overlays for each frame and BPM text.
    Returns a list of BGR overlay images to place on the output video.
    """
    n = len(signal)
    if n == 0:
        return []
    width, height = size
    # Normalize to [0, 1]
    smin, smax = float(signal.min()), float(signal.max())
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
    norm = (signal - smin) / denom
    ys = (height - 20) - (height - 40) * norm
    xs_full = np.linspace(10, width - 10, n)
    points_full = np.vstack([xs_full, ys]).T.astype(np.int32)

    overlays: List[np.ndarray] = []
    for i in range(n):
        overlay = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (200, 200, 200), 1)
        # Draw waveform up to i
        end_idx = max(2, i + 1)
        pts = points_full[:end_idx]
        for j in range(1, len(pts)):
            cv2.line(overlay, tuple(pts[j-1]), tuple(pts[j]), (60, 60, 200), 2)
        # Cursor
        xcur = int(xs_full[i])
        cv2.line(overlay, (xcur, 10), (xcur, height - 10), (0, 0, 255), 1)
        # BPM text
        cv2.putText(overlay, f"BPM: {int(round(bpm))}", (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        overlays.append(overlay)
    return overlays