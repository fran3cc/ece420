import numpy as np

def extract_green_signal(frames, roi_fn):
    values = []
    for frame in frames:
        x, y, w, h = roi_fn(frame)
        roi = frame[y:y+h, x:x+w]
        g = roi[:, :, 1]
        values.append(float(np.mean(g)))
    return np.asarray(values, dtype=np.float64)

def pos_signal_from_rgb(rgb, fps):
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return np.zeros((len(rgb),), dtype=np.float64)
    m = np.mean(rgb, axis=0)
    m[m == 0] = 1.0
    n = rgb / m - 1.0
    r = n[:, 2]
    g = n[:, 1]
    b = n[:, 0]
    x = 3.0*r - 2.0*g
    y = 1.5*r + g - 1.5*b
    sx = np.std(x) if len(x) > 0 else 1.0
    sy = np.std(y) if len(y) > 0 else 1.0
    alpha = sx/(sy + 1e-8)
    s = x + alpha*y
    return s