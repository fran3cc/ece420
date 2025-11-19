import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from rppg import VideoReader, make_dynamic_face_forehead_roi, extract_green_signal, detrend_and_bandpass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='outputs/filtered_signal.csv')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vr = VideoReader.from_file(args.video)
    roi = make_dynamic_face_forehead_roi(update_every=10, smooth=0.7)
    sig = extract_green_signal(vr.frames(), roi)
    y = detrend_and_bandpass(sig, vr.fps)
    np.savetxt(args.out, y, delimiter=',')
    print(args.out)
    vr.release()

if __name__ == '__main__':
    main()