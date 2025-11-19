import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rppg import VideoReader, make_dynamic_face_forehead_roi, extract_green_signal, detrend_and_bandpass, estimate_hr_bpm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    args = parser.parse_args()
    vr = VideoReader.from_file(args.video)
    roi = make_dynamic_face_forehead_roi(update_every=10, smooth=0.7)
    sig = extract_green_signal(vr.frames(), roi)
    y = detrend_and_bandpass(sig, vr.fps)
    bpm = estimate_hr_bpm(y, vr.fps)
    print(int(round(bpm)))
    vr.release()

if __name__ == '__main__':
    main()