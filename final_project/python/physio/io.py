"""
Video I/O utilities for reading frames and saving results.
"""

import cv2
import numpy as np
import csv
import json
from pathlib import Path
from typing import Optional, List
from .types import FrameBatch, SignalSeries, Spectrum, RateEstimate


def read_video_frames(video_path: str, max_frames: Optional[int] = None) -> FrameBatch:
    """
    Read video frames from file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None for all)
        
    Returns:
        FrameBatch with frames and FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default fallback
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame.copy())
        frame_count += 1
        
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    
    return FrameBatch(frames_bgr=frames, fps=fps)


def save_signal_csv(signal: SignalSeries, output_path: str) -> None:
    """
    Save signal to CSV file.
    
    Args:
        signal: Signal series to save
        output_path: Output CSV file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_sec', 'value'])
        
        for i, value in enumerate(signal.values):
            time_sec = i / signal.fps
            writer.writerow([time_sec, value])


def save_spectrum_csv(spectrum: Spectrum, output_path: str) -> None:
    """
    Save spectrum to CSV file.
    
    Args:
        spectrum: Spectrum to save
        output_path: Output CSV file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['freq_hz', 'magnitude'])
        
        for freq, mag in zip(spectrum.freqs_hz, spectrum.magnitude):
            writer.writerow([freq, mag])


def save_results_json(rate_estimate: RateEstimate, output_path: str, 
                     metadata: Optional[dict] = None) -> None:
    """
    Save rate estimate and metadata to JSON file.
    
    Args:
        rate_estimate: Rate estimate to save
        output_path: Output JSON file path
        metadata: Additional metadata to include
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'rate': rate_estimate.rate,
        'freq_hz': rate_estimate.freq_hz,
        'metadata': metadata or {}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


class LiveVideoCapture:
    """Live video capture for real-time processing."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize live video capture.
        
        Args:
            device_id: Camera device ID (0 for default camera)
        """
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera device {device_id}")
        
        # Try to get actual FPS, fallback to 30
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame.
        
        Returns:
            Frame in BGR format, or None if failed
        """
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self) -> None:
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()