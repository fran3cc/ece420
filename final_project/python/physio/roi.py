"""
Region of Interest (ROI) selection for pulse and respiration detection.
"""

from typing import Protocol, Tuple
import numpy as np


class ROISelector(Protocol):
    """Protocol for ROI selection strategies."""
    
    def select(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Select ROI from a frame.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Tuple of (x, y, w, h) defining the ROI rectangle
        """
        ...


class FaceROI:
    """Simple heuristic-based face ROI selector for pulse detection."""
    
    def select(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Select forehead region for pulse detection.
        Uses simple heuristics - can be replaced with face detection later.
        """
        h, w = frame_bgr.shape[:2]
        
        # Simple forehead region: upper-center portion of frame
        x = w // 4
        y = h // 6
        roi_w = w // 2
        roi_h = h // 5
        
        # Ensure ROI is within frame bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = max(1, min(roi_w, w - x))
        roi_h = max(1, min(roi_h, h - y))
        
        return x, y, roi_w, roi_h


class ChestROI:
    """Simple heuristic-based chest ROI selector for respiration detection."""
    
    def select(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Select chest region for respiration detection.
        Uses simple heuristics - can be replaced with pose detection later.
        """
        h, w = frame_bgr.shape[:2]
        
        # Simple chest region: center portion of frame
        x = w // 4
        y = h // 2
        roi_w = w // 2
        roi_h = h // 3
        
        # Ensure ROI is within frame bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = max(1, min(roi_w, w - x))
        roi_h = max(1, min(roi_h, h - y))
        
        return x, y, roi_w, roi_h