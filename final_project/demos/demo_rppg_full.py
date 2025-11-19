import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from rppg import VideoReader, make_dynamic_face_forehead_roi, extract_green_signal, detrend_and_bandpass, estimate_hr_bpm

def run(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vr = VideoReader.from_file(video_path)
    roi = make_dynamic_face_forehead_roi(update_every=10, smooth=0.7)
    sig = extract_green_signal(vr.frames(), roi)
    y = detrend_and_bandpass(sig, vr.fps)
    bpm = estimate_hr_bpm(y, vr.fps)
    np.savetxt(os.path.join(out_dir, 'raw_signal.csv'), sig, delimiter=',')
    np.savetxt(os.path.join(out_dir, 'filtered_signal.csv'), y, delimiter=',')
    with open(os.path.join(out_dir, 'heart_rate.txt'), 'w') as f:
        f.write(str(int(round(bpm))))
    print(os.path.join(out_dir, 'raw_signal.csv'))
    print(os.path.join(out_dir, 'filtered_signal.csv'))
    print(os.path.join(out_dir, 'heart_rate.txt'))
    vr.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='outputs/full')
    args = parser.parse_args()
    run(args.video, args.out)

if __name__ == '__main__':
    main()