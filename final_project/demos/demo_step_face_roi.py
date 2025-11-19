import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from rppg import VideoReader, detect_face_roi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='outputs/roi.png')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vr = VideoReader.from_file(args.video)
    import cv2
    first = None
    for f in vr.frames():
        first = f
        break
    x, y, w, h = detect_face_roi(first)
    cv2.rectangle(first, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(args.out, first)
    print(args.out)
    vr.release()

if __name__ == '__main__':
    main()