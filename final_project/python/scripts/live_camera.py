#!/usr/bin/env python3
"""Live camera demo for real-time physiological signal detection with video recording.
Enhanced version with stable rate display and video recording for offline processing."""

import argparse
import time
import sys
import os
from collections import deque
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from physio.roi import FaceROI, ChestROI, ROISelector
from physio.filters import moving_average_detrend, butter_bandpass
from physio.spectrum import fft_spectrum, pick_peak, hz_to_bpm, hz_to_breaths
from physio.config import (PULSE_BAND_HZ, RESP_BAND_HZ, LIVE_WINDOW_SEC, 
                          LIVE_UPDATE_INTERVAL_SEC, EMA_SMOOTHING_FACTOR)


def parse_args():
    parser = argparse.ArgumentParser(description="Live physiological signal detection with video recording")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--mode", choices=["pulse", "respiration"], default="pulse",
                        help="Detection mode")
    parser.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Frequency band in Hz (overrides mode default)")
    parser.add_argument("--window-sec", type=float, default=LIVE_WINDOW_SEC,
                        help="Rolling window size in seconds")
    parser.add_argument("--update-interval", type=float, default=LIVE_UPDATE_INTERVAL_SEC,
                        help="Rate computation update interval in seconds")
    parser.add_argument("--ema", type=float, default=EMA_SMOOTHING_FACTOR,
                        help="EMA smoothing factor for rate estimates")
    parser.add_argument("--output-dir", type=str, default="recordings",
                        help="Directory to save recorded videos")
    return parser.parse_args()


def compute_luma(roi_bgr: np.ndarray) -> float:
    """Compute ITU-R BT.601 luma from BGR ROI."""
    if roi_bgr.size == 0:
        return 0.0
    return (0.114 * roi_bgr[:, :, 0].mean() + 
            0.587 * roi_bgr[:, :, 1].mean() + 
            0.299 * roi_bgr[:, :, 2].mean())


class VideoRecorder:
    """Video recorder for capturing physiological signal data."""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.writer = None
        self.recording = False
        self.filename = None
        
    def start_recording(self, frame_width, frame_height, fps=30.0):
        """Start video recording."""
        if self.recording:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.output_dir / f"physio_recording_{timestamp}.mp4"
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.filename), fourcc, fps, (frame_width, frame_height))
        
        if self.writer.isOpened():
            self.recording = True
            return True
        else:
            self.writer = None
            return False
    
    def write_frame(self, frame):
        """Write a frame to the video file."""
        if self.recording and self.writer is not None:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop video recording."""
        if self.recording and self.writer is not None:
            self.writer.release()
            self.writer = None
            self.recording = False
            return self.filename
        return None
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording



def draw_roi_overlay(frame: np.ndarray, roi_coords: tuple, mode: str, 
                    current_rate: str, stable_rate: str, fps_str: str, 
                    recording_status: str) -> None:
    """Draw ROI rectangle and information overlay on frame."""
    x, y, w, h = roi_coords
    
    # Draw ROI rectangle
    color = (0, 255, 0) if mode == "pulse" else (255, 0, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw mode label on ROI
    label = "FACE" if mode == "pulse" else "CHEST"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 2, cv2.LINE_AA)
    
    # Create stable display sections
    info_y = 30
    
    # Current detection (may fluctuate)
    cv2.putText(frame, f"Current {mode.title()}: {current_rate}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 1, cv2.LINE_AA)
    
    # Stable rate (smoothed, less fluctuation)
    cv2.putText(frame, f"Stable Rate: {stable_rate}", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2, cv2.LINE_AA)
    
    # FPS and recording status
    cv2.putText(frame, fps_str, (10, info_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Recording status
    rec_color = (0, 0, 255) if "Recording" in recording_status else (150, 150, 150)
    cv2.putText(frame, recording_status, (10, info_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2, cv2.LINE_AA)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit, 's' to switch mode, 'r' to record", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Cannot open camera device {args.device}")
        return 1
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Starting live {args.mode} detection with video recording...")
    print("Press 'q' to quit, 's' to switch modes, 'r' to start/stop recording")
    
    # Initialize processing parameters
    current_mode = args.mode
    roi_selector: ROISelector = FaceROI() if current_mode == "pulse" else ChestROI()
    band = args.band or (PULSE_BAND_HZ if current_mode == "pulse" else RESP_BAND_HZ)
    
    # Initialize video recorder
    recorder = VideoRecorder(args.output_dir)
    
    # Rolling window for signal values and timestamps
    values = deque()
    times = deque()
    
    # Rate tracking with separate current and stable displays
    current_rate = None
    stable_rate = None
    ema_rate = None
    last_compute_time = 0.0
    
    # Rate history for stability
    rate_history = deque(maxlen=10)
    
    # FPS calculation
    fps_counter = deque(maxlen=30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            current_time = time.time()
            
            # Calculate FPS
            fps_counter.append(current_time)
            if len(fps_counter) > 1:
                fps_estimate = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])
            else:
                fps_estimate = 30.0  # Default assumption
            
            # Record frame if recording
            if recorder.is_recording():
                recorder.write_frame(frame)
            
            # Get ROI
            x, y, w, h = roi_selector.select(frame)
            
            # Ensure ROI is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            roi_img = frame[y:y+h, x:x+w]
            
            # Extract signal value
            if current_mode == "pulse":
                if roi_img.size > 0:
                    val = float(roi_img[:, :, 1].mean())  # Green channel
                else:
                    val = 0.0
            else:  # respiration
                val = compute_luma(roi_img)
            
            # Add to rolling window
            values.append(val)
            times.append(current_time)
            
            # Trim window to specified duration
            while times and (times[-1] - times[0] > args.window_sec):
                values.popleft()
                times.popleft()
            
            # Initialize display strings
            current_rate_str = "Initializing..."
            stable_rate_str = "Waiting for data..."
            fps_str = f"FPS: {fps_estimate:.1f}"
            
            # Recording status
            if recorder.is_recording():
                recording_status = "● Recording"
            else:
                recording_status = "○ Not Recording"
            
            # Reduced minimum requirements for faster startup
            min_frames = max(15, int(1.5 * fps_estimate))  # Even more reduced
            min_duration = 2.0  # Reduced to 2 seconds
            
            # Compute rate estimate
            if (fps_estimate and 
                (current_time - last_compute_time) > args.update_interval and
                len(values) >= min_frames and
                len(times) > 1 and
                (times[-1] - times[0]) >= min_duration):
                
                try:
                    # Convert to numpy array
                    signal_array = np.array(values, dtype=np.float32)
                    
                    # Signal processing pipeline with adaptive parameters
                    window_sec = max(1.5, min(3.0, (times[-1] - times[0]) / 2))  # More adaptive
                    detrended = moving_average_detrend(signal_array, fps_estimate, window_sec)
                    filtered = butter_bandpass(detrended, fps_estimate, band[0], band[1])
                    
                    # Frequency analysis
                    freqs, magnitude = fft_spectrum(filtered, fps_estimate)
                    f_peak = pick_peak(freqs, magnitude, band[0], band[1])
                    
                    # Convert to rate
                    if current_mode == "pulse":
                        rate = hz_to_bpm(f_peak)
                        unit = "BPM"
                    else:
                        rate = hz_to_breaths(f_peak)
                        unit = "breaths/min"
                    
                    # Update current rate (may fluctuate)
                    if rate > 0:
                        current_rate = rate
                        current_rate_str = f"{rate:.1f} {unit}"
                        
                        # Add to rate history for stable display
                        rate_history.append(rate)
                        
                        # Apply EMA smoothing for stable rate
                        if ema_rate is None:
                            ema_rate = rate
                        else:
                            ema_rate = args.ema * rate + (1 - args.ema) * ema_rate
                        
                        # Only update stable rate if we have enough history
                        if len(rate_history) >= 3:
                            stable_rate = ema_rate
                            stable_rate_str = f"{ema_rate:.1f} {unit}"
                    else:
                        current_rate_str = "No peak detected"
                    
                    last_compute_time = current_time
                    
                except Exception as e:
                    current_rate_str = "Processing error"
            
            # Keep stable rate display even when not computing
            if stable_rate is not None:
                unit = "BPM" if current_mode == "pulse" else "breaths/min"
                stable_rate_str = f"{stable_rate:.1f} {unit}"
            
            # Draw overlay with separate current and stable displays
            draw_roi_overlay(frame, (x, y, w, h), current_mode, 
                           current_rate_str, stable_rate_str, fps_str, recording_status)
            
            # Display frame
            cv2.imshow("Physio Live Demo - Rate Detection & Recording", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch mode
                current_mode = "respiration" if current_mode == "pulse" else "pulse"
                roi_selector = FaceROI() if current_mode == "pulse" else ChestROI()
                band = PULSE_BAND_HZ if current_mode == "pulse" else RESP_BAND_HZ
                
                # Reset state
                values.clear()
                times.clear()
                current_rate = None
                stable_rate = None
                ema_rate = None
                rate_history.clear()
                last_compute_time = 0.0
                
                print(f"Switched to {current_mode} mode")
            elif key == ord('r'):
                # Toggle recording
                if recorder.is_recording():
                    filename = recorder.stop_recording()
                    print(f"Recording stopped. Saved to: {filename}")
                else:
                    if recorder.start_recording(frame_width, frame_height, fps_estimate):
                        print("Recording started...")
                    else:
                        print("Failed to start recording")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop recording if active
        if recorder.is_recording():
            filename = recorder.stop_recording()
            print(f"Recording stopped. Saved to: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())