import numpy as np

def default_center_roi(frame):
    h, w = frame.shape[:2]
    rw, rh = int(w*0.3), int(h*0.2)
    x = (w - rw)//2
    y = (h - rh)//3
    return (x, y, rw, rh)

class _DynamicForeheadROI:
    def __init__(self, update_every=10, smooth=0.7):
        self.update_every = max(1, int(update_every))
        self.smooth = float(smooth)
        self.face_cascade = None
        self.t = 0
        self.last_face = None
    def _detect(self, frame):
        import cv2
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2]*f[3])
    def __call__(self, frame):
        h, w = frame.shape[:2]
        need_update = (self.t % self.update_every == 0) or (self.last_face is None)
        face = None
        if need_update:
            face = self._detect(frame)
            if face is not None:
                if self.last_face is None:
                    self.last_face = np.array(face, dtype=np.float64)
                else:
                    self.last_face = self.smooth*np.array(face, dtype=np.float64) + (1.0-self.smooth)*self.last_face
        self.t += 1
        if self.last_face is None:
            return default_center_roi(frame)
        fx, fy, fw, fh = [int(round(v)) for v in self.last_face]
        fx = max(0, min(fx, w-1))
        fy = max(0, min(fy, h-1))
        fw = max(1, min(fw, w-fx))
        fh = max(1, min(fh, h-fy))
        rx = fx + int(fw*0.25)
        ry = fy + int(fh*0.10)
        rw = int(fw*0.50)
        rh = int(fh*0.22)
        rx = max(0, min(rx, w-1))
        ry = max(0, min(ry, h-1))
        rw = max(1, min(rw, w-rx))
        rh = max(1, min(rh, h-ry))
        return (rx, ry, rw, rh)

def detect_face_roi(frame):
    try:
        import cv2
    except Exception:
        return default_center_roi(frame)
    tracker = _DynamicForeheadROI()
    return tracker(frame)

def make_dynamic_face_forehead_roi(update_every=10, smooth=0.7):
    return _DynamicForeheadROI(update_every=update_every, smooth=smooth)