import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rppg import VideoReader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    args = parser.parse_args()
    vr = VideoReader.from_file(args.video)
    c = 0
    for _ in vr.frames():
        c += 1
    print(c)
    vr.release()

if __name__ == '__main__':
    main()