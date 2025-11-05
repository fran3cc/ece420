# Eulerian Video Magnification (EVM)

Simple Python CLI that matches the MATLAB Gaussian+Ideal pipeline and prints pulse BPM in the terminal from two methods:
- Color-based pulse (green channel, measured on the EVM output video)
- Motion-based pulse (head motion via optical flow + PCA on original video)

The output video stays the same style as the MATLAB results.

## Quick Start

Process a video using a predefined config:

```bash
python main.py --video final_project/python/samples/baby2.mp4 --config baby2
```

Process a custom video with manual parameters:

```bash
python main.py \
  --video /path/to/input.mp4 \
  --alpha 50 \
  --levels 4 \
  --fl 0.9 \
  --fh 1.3 \
  --chrom-attenuation 1.0
```

Output:
- Saves `/<path>/<name>_evm.mp4`
- Prints two lines in terminal:
  - `Color BPM (EVM video): <value>`
  - `Motion BPM (optical flow): <value>`

## Picking Parameters (student-style guide)

Frequency band is the most important. It’s in Hz, BPM = 60 × Hz.

- Baby pulse (high HR 120–180 BPM):
  - Use `--fl 2.0 --fh 3.0` (Hz)
  - Example: `--alpha 150 --levels 6 --fl 140/60 --fh 160/60` (matches baby2 MATLAB)
- Adult face pulse (normal HR 50–100 BPM):
  - Use `--fl 0.8 --fh 1.7` (Hz)
- Motion amplification (not pulse, e.g., vibrations):
  - Use higher band like `--fl 3 --fh 10` (Hz)

Other knobs:
- `--alpha` (gain): start at 50. If color changes are faint, try 80–150. If shadows/noise appear, lower it.
- `--levels` (Gaussian downsample levels): usually 4–6. Higher levels blur more before magnification.
- `--chrom-attenuation`: 1.0 keeps chroma neutral. 1.5–2.5 can boost color changes.

Tips:
- If color BPM looks wrong, check the band. Adult videos need ~0.8–1.7 Hz; babies need ~2.0–3.0 Hz.
- If motion BPM reads breathing instead of pulse, the band is too low. Raise to baby pulse range for infants.
- Face-crops help. You can crop first with ffmpeg:
  ```bash
  ffmpeg -i input.mp4 -vf "crop=W:H:X:Y" -c:a copy input_facecrop.mp4
  ```

## Usage Examples

Predefined configs (auto parameters):
- Baby (Gaussian+Ideal):
  ```bash
  python main.py --video final_project/python/samples/baby2.mp4 --config baby2
  ```
- Adult face (Gaussian+Ideal):
  ```bash
  python main.py --video final_project/python/samples/face.mp4 --config face
  ```

Manual tuning:
- Adult face (normal HR):
  ```bash
  python main.py --video /path/to/face.mp4 --alpha 50 --levels 4 --fl 0.9 --fh 1.3 --chrom-attenuation 1.0
  ```
- Baby (high HR):
  ```bash
  python main.py --video /path/to/baby.mp4 --alpha 150 --levels 6 --fl 2.2 --fh 2.6 --chrom-attenuation 1.0
  ```

## What’s inside

- Gaussian pyramid + ideal temporal bandpass (matches MATLAB `amplify_spatial_Gdown_temporal_ideal.m`)
- NTSC color space with optional chroma amplification
- Two BPM results printed in terminal: color (from magnified video) and motion (optical flow)

Predefined configs live in `physio/config.py`. Only Gaussian+Ideal is implemented in Python (Laplacian variants are MATLAB-only).