import os
import numpy as np

class VideoReader:
    def __init__(self, cap, fps):
        self.cap = cap
        self._fps = fps
    @staticmethod
    def from_file(path):
        try:
            import cv2
        except Exception:
            raise RuntimeError("OpenCV is required")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return VideoReader(cap, float(fps))
    @staticmethod
    def from_camera(index=0):
        try:
            import cv2
        except Exception:
            raise RuntimeError("OpenCV is required")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return VideoReader(cap, float(fps))
    @property
    def fps(self):
        return self._fps
    def frames(self):
        import cv2
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame
    def release(self):
        self.cap.release()