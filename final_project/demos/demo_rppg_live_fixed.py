import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from collections import deque
import numpy as np
import cv2
from rppg import VideoReader, make_dynamic_face_forehead_roi, detrend_and_bandpass, estimate_hr_bpm, pos_signal_from_rgb

def sparkline(values, size=(300, 120)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if len(values) < 2:
        return img
    v = np.asarray(values, dtype=np.float64)
    v = v - np.mean(v)
    if np.std(v) > 1e-9:
        v = v / (np.std(v) * 3.0)
    v = np.clip(v, -1, 1)
    y = (h//2) - (v * (h//2 - 10))
    x = np.linspace(0, w-1, num=len(v))
    pts = np.stack([x, y], axis=1).astype(np.int32)
    for i in range(1, len(pts)):
        cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 0), 2)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--min_window', type=float, default=15.0)
    parser.add_argument('--max_window', type=float, default=30.0)
    args = parser.parse_args()

    vr = VideoReader.from_camera(args.camera)
    cap = vr.cap

    times = deque()
    raw_rgb = deque()
    filtered_values = []
    bpm = None
    last_update = 0.0

    roi_fn = make_dynamic_face_forehead_roi(update_every=10, smooth=0.7)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        x, y, w, h = roi_fn(frame)
        roi_patch = frame[y:y+h, x:x+w]
        if roi_patch.size > 0:
            r = float(np.mean(roi_patch[:, :, 2]))
            g = float(np.mean(roi_patch[:, :, 1]))
            b = float(np.mean(roi_patch[:, :, 0]))
        else:
            r = g = b = 0.0
        times.append(t)
        raw_rgb.append((r, g, b))
        while len(times) > 1 and (times[-1] - times[0]) > args.max_window:
            times.popleft()
            raw_rgb.popleft()
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        fps_est = (len(raw_rgb) / duration) if duration > 0.1 else vr.fps
        show_text = "Measuring..."
        rgb = np.asarray(raw_rgb, dtype=np.float64)
        if duration >= args.min_window and fps_est and 10.0 <= fps_est <= 120.0:
            if (t - last_update) > 1.0:
                ysig = pos_signal_from_rgb(rgb, fps_est)
                ysig = detrend_and_bandpass(ysig, fps_est)
                bpm = estimate_hr_bpm(ysig, fps_est)
                filtered_values = ysig.tolist()
                last_update = t
            show_text = f"HR: {int(round(bpm))} BPM" if bpm and bpm > 0 else "HR: --"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, show_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        fill_ratio = min(1.0, duration / args.min_window) if args.min_window > 0 else 1.0
        bar_w = int(200 * fill_ratio)
        cv2.rectangle(frame, (20, 60), (220, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 60), (20+bar_w, 80), (0, 255, 0), -1)
        sig_img = sparkline(filtered_values if filtered_values else (rgb[:,1] if len(rgb)>0 else []))
        cv2.imshow('rPPG Live', frame)
        cv2.imshow('Signal', sig_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            out_dir = os.path.join('outputs', 'live')
            os.makedirs(out_dir, exist_ok=True)
            if len(rgb) > 0:
                np.savetxt(os.path.join(out_dir, 'raw_signal.csv'), rgb, delimiter=',')
            if filtered_values:
                np.savetxt(os.path.join(out_dir, 'filtered_signal.csv'), np.asarray(filtered_values), delimiter=',')
            if bpm and bpm > 0:
                with open(os.path.join(out_dir, 'heart_rate.txt'), 'w') as f:
                    f.write(str(int(round(bpm))))
            print(out_dir)

    vr.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()