#!/usr/bin/env python3
"""
Simplified EVM CLI: keep MATLAB-compatible video generation and print BPMs.
- Processes video with Gaussian+Ideal EVM
- Prints color-based BPM (from EVM video) and motion-based BPM (optical flow)
- Optional --auto-detect to choose best BPM band (adult/child/baby)
"""
import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from physio.config import VIDEO_CONFIGS, PULSE_BAND_HZ
from physio.io import read_video_frames
from physio.roi import FaceROI
from physio.filters import moving_average_detrend, butter_bandpass
from physio.spectrum import fft_spectrum, pick_peak, hz_to_bpm
from physio.motion_pulse_detector import (
    track_head_motion_optical_flow,
    extract_pulse_from_motion,
)

# --- NTSC helpers ---
def rgb_to_ntsc(rgb):
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    I = 0.596*R - 0.274*G - 0.322*B
    Q = 0.211*R - 0.523*G + 0.312*B
    return np.stack([Y, I, Q], axis=-1)

def ntsc_to_rgb(nt):
    Y, I, Q = nt[...,0], nt[...,1], nt[...,2]
    R = Y + 0.956*I + 0.621*Q
    G = Y - 0.272*I - 0.647*Q
    B = Y - 1.105*I + 1.702*Q
    return np.stack([R, G, B], axis=-1)

# --- Core EVM (Gaussian downsample + Ideal bandpass) ---
def amplify_spatial_gdown_temporal_ideal(video_path, output_path, alpha, level, fl, fh, chrom_attenuation=1.0, mask_face=False):
    batch = read_video_frames(video_path)
    frames = batch.frames_bgr
    fps = batch.fps
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to process")

    # Build G-down stack in NTSC at target level
    gstack = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for _ in range(max(0, level-1)):
            rgb = cv2.pyrDown(rgb)
        ntsc = rgb_to_ntsc(rgb)
        gstack.append(ntsc)
    gstack = np.stack(gstack, axis=0)  # (t,h,w,3)

    # Temporal ideal bandpass via FFT along time axis
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    mask = (freqs >= float(fl)) & (freqs <= float(fh))
    F = np.fft.rfft(gstack, axis=0)
    F[~mask, ...] = 0
    filtered = np.fft.irfft(F, n=n, axis=0)

    # Amplify channels
    filtered[...,0] *= alpha
    filtered[...,1] *= alpha * chrom_attenuation
    filtered[...,2] *= alpha * chrom_attenuation

    # Render to output video
    h0, w0 = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    # Build smooth face mask at full resolution (if enabled)
    face_mask3 = None
    if mask_face:
        roi = FaceROI()
        x, y, rw, rh = roi.select(frames[0])
        face_mask = np.zeros((h0, w0), dtype=np.float32)
        face_mask[y:y+rh, x:x+rw] = 1.0
        # Smooth edges to avoid hard boundaries
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), 7)
        face_mask3 = np.repeat(face_mask[..., None], 3, axis=-1)

    for i in range(n):
        base_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        small = filtered[i]
        up = cv2.resize(small, (w0, h0), interpolation=cv2.INTER_LINEAR)
        if face_mask3 is not None:
            up = up * face_mask3
        ntsc = rgb_to_ntsc(base_rgb) + up
        rgb = np.clip(ntsc_to_rgb(ntsc), 0.0, 1.0)
        bgr_out = cv2.cvtColor((rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr_out)
    out.release()
    return output_path

# --- BPM computations ---
def peak_and_snr(freqs: np.ndarray, magnitude: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
    """Return (f_peak_hz, snr) where snr = peak_mag / median_band_mag."""
    fmin, fmax = band
    if len(freqs) == 0 or len(magnitude) == 0:
        return 0.0, 0.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0
    band_freqs = freqs[mask]
    band_mag = magnitude[mask]
    f_peak = pick_peak(freqs, magnitude, fmin, fmax)
    if f_peak <= 0:
        return 0.0, 0.0
    idx = (np.abs(band_freqs - f_peak)).argmin()
    peak_mag = float(band_mag[idx])
    noise = float(np.median(band_mag)) + 1e-8
    snr = peak_mag / noise
    return float(f_peak), float(snr)


def calculate_color_pulse_metrics(frames_bgr, fps, band: tuple[float, float]):
    roi = FaceROI()
    x, y, w, h = roi.select(frames_bgr[0])
    vals = []
    for f in frames_bgr:
        roi_img = f[y:y+h, x:x+w]
        vals.append(float(roi_img[..., 1].mean()))
    sig = np.asarray(vals, dtype=np.float32)
    detr = moving_average_detrend(sig, fps, 4.0)
    filt = butter_bandpass(detr, fps, band[0], band[1])
    freqs, mag = fft_spectrum(filt, fps)
    fpk, snr = peak_and_snr(freqs, mag, band)
    bpm = hz_to_bpm(fpk)
    return float(bpm), float(snr)


def calculate_motion_pulse_metrics(frames_bgr, fps, band: tuple[float, float]):
    traces, _ = track_head_motion_optical_flow(frames_bgr, fps)
    comp, bpm = extract_pulse_from_motion(traces, fps, band)
    # Compute SNR on selected component
    if comp.size == 0:
        return float(bpm), 0.0
    freqs, mag = fft_spectrum(comp.astype(np.float64), fps)
    fpk, snr = peak_and_snr(freqs, mag, band)
    return float(bpm), float(snr)

# --- Config helpers ---
def get_video_name(path):
    return Path(path).stem

def load_config(config_name=None, video_path=None):
    if config_name:
        return VIDEO_CONFIGS.get(config_name)
    if video_path:
        return VIDEO_CONFIGS.get(get_video_name(video_path))
    return None

# --- Confidence metric ---
def bpm_confidence(color_bpm: float, motion_bpm: float) -> float:
    if color_bpm <= 0 or motion_bpm <= 0:
        return 0.0
    denom = max(color_bpm, motion_bpm, 1e-6)
    diff = abs(color_bpm - motion_bpm)
    return max(0.0, 1.0 - diff / denom)

# --- CLI ---
def main():
    p = argparse.ArgumentParser(description='EVM (Gaussian+Ideal) with terminal BPM output')
    p.add_argument('--video', '-v', required=True, help='Input video file')
    p.add_argument('--output', '-o', help='Output video file (default: <name>_evm.mp4)')
    p.add_argument('--config', '-c', help='Predefined config name (e.g., baby2, face)')
    p.add_argument('--alpha', type=float)
    p.add_argument('--levels', type=int)
    p.add_argument('--fl', type=float)
    p.add_argument('--fh', type=float)
    p.add_argument('--chrom-attenuation', type=float)
    p.add_argument('--mask-face', action='store_true', help='Limit magnification to face region only')
    p.add_argument('--adaptive-motion', action='store_true', help='Use color BPM to set motion band (±5 BPM)')
    # Preset bands (simple, reliable)
    p.add_argument('--adult', '--adults', dest='adult', action='store_true', help='Adult resting range (0.8–1.7 Hz)')
    p.add_argument('--child', '--children', dest='child', action='store_true', help='Child resting range (1.3–2.3 Hz)')
    p.add_argument('--baby', dest='baby', action='store_true', help='Baby resting range (2.0–3.0 Hz)')
    p.add_argument('--exercise', action='store_true', help='Adult exercise range (1.5–3.5 Hz)')
    p.add_argument('--medical', action='store_true', help='Medical wide range (0.5–4.0 Hz)')
    args = p.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        return 1

    cfg = load_config(args.config, args.video) or {}
    alpha = args.alpha if args.alpha is not None else cfg.get('alpha', 50)
    levels = args.levels if args.levels is not None else cfg.get('levels', 4)
    fl = args.fl if args.fl is not None else cfg.get('fl', PULSE_BAND_HZ[0])
    fh = args.fh if args.fh is not None else cfg.get('fh', PULSE_BAND_HZ[1])
    chrom = args.chrom_attenuation if args.chrom_attenuation is not None else cfg.get('chrom_attenuation', 1.0)

    # Preset selection (presets override manual params). Only one preset allowed.
    preset_flags = {
        'adult': args.adult,
        'child': args.child,
        'baby': args.baby,
        'exercise': args.exercise,
        'medical': args.medical,
    }
    preset_count = sum(1 for v in preset_flags.values() if v)
    if preset_count > 1:
        print('Error: choose only one preset among --adult/--child/--baby/--exercise/--medical')
        return 1
    preset_name = None
    if args.adult:
        fl, fh = 0.8, 1.7; preset_name = 'Adult'
    elif args.child:
        fl, fh = 1.3, 2.3; preset_name = 'Child'
    elif args.baby:
        fl, fh = 2.0, 3.0; preset_name = 'Baby'
    elif args.exercise:
        fl, fh = 1.5, 3.5; preset_name = 'Adult Exercise'
    elif args.medical:
        fl, fh = 0.5, 4.0; preset_name = 'Medical (wide)'

    in_path = Path(args.video)
    out_path = args.output or str(in_path.parent / f"{in_path.stem}_evm.mp4")

    # Single processing path (no auto-detect)
    band = (float(fl), float(fh))
    print(f"Processing: {args.video}")
    if preset_name:
        print(f"Preset: {preset_name} ({band[0]:.2f}-{band[1]:.2f} Hz)")
    print(f"Params: alpha={alpha}, levels={levels}, fl={band[0]:.3f}Hz, fh={band[1]:.3f}Hz, chrom={chrom}")
    amplify_spatial_gdown_temporal_ideal(args.video, out_path, alpha, levels, band[0], band[1], chrom, args.mask_face)
    print(f"EVM video saved: {out_path}")

    # Compute BPMs
    evm_batch = read_video_frames(out_path)
    color_bpm, color_snr = calculate_color_pulse_metrics(evm_batch.frames_bgr, evm_batch.fps, band)
    orig_batch = read_video_frames(args.video)
    mot_band = band
    if args.adaptive_motion and color_bpm > 0:
        center_hz = color_bpm / 60.0
        # Widen band to ±10 BPM and add small random offset (±2 BPM)
        offset_bpm = float(np.random.uniform(-2.0, 2.0))
        center_hz = center_hz + (offset_bpm / 60.0)
        delta_hz = 10.0 / 60.0
        mot_fl = max(0.3, center_hz - delta_hz)
        mot_fh = min(4.0, center_hz + delta_hz)
        mot_band = (mot_fl, mot_fh)
        print(f"Adaptive motion band: {mot_band[0]:.3f}-{mot_band[1]:.3f} Hz (color {color_bpm:.2f} BPM, offset {offset_bpm:+.2f} BPM)")
    motion_bpm, motion_snr = calculate_motion_pulse_metrics(orig_batch.frames_bgr, orig_batch.fps, mot_band)

    print("-"*60)
    print(f"Color BPM (EVM video): {color_bpm:.2f}")
    print(f"Motion BPM (optical flow): {motion_bpm:.2f}")
    if args.adaptive_motion:
        print(f"Motion band used: {mot_band[0]:.3f}-{mot_band[1]:.3f} Hz")
    print("-"*60)
    return 0

if __name__ == '__main__':
    sys.exit(main())