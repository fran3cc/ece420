#!/usr/bin/env python3
"""
Eulerian Video Magnification - Python implementation matching MATLAB original
Based on: "Eulerian Video Magnification for Revealing Subtle Changes in the World"
ACM Transaction on Graphics, Volume 31, Number 4 (Proceedings SIGGRAPH 2012)

This implementation exactly matches the MATLAB amplify_spatial_Gdown_temporal_ideal function.
"""

import argparse
import sys
import numpy as np
import cv2
import tqdm
from pathlib import Path

# Add parent directory to path for physio imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from physio.config import PULSE_BAND_HZ, RESP_BAND_HZ

def binomial_filter(size):
    """Generate binomial filter coefficients (matches MATLAB binomialFilter)."""
    if size < 2:
        raise ValueError("Size must be larger than 1")
    
    kernel = np.array([0.5, 0.5])
    for _ in range(size - 2):
        kernel = np.convolve([0.5, 0.5], kernel)
    
    return kernel

def rgb2ntsc(rgb_image):
    """Convert RGB to NTSC color space (matches MATLAB rgb2ntsc)."""
    # NTSC conversion matrix (same as MATLAB)
    transform_matrix = np.array([
        [0.299,  0.587,  0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523,  0.312]
    ])
    
    # Reshape for matrix multiplication
    original_shape = rgb_image.shape
    rgb_flat = rgb_image.reshape(-1, 3)
    
    # Apply transformation
    ntsc_flat = rgb_flat @ transform_matrix.T
    
    return ntsc_flat.reshape(original_shape)

def ntsc2rgb(ntsc_image):
    """Convert NTSC to RGB color space (matches MATLAB ntsc2rgb)."""
    # Inverse NTSC conversion matrix
    transform_matrix = np.array([
        [1.0,  0.956,  0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106,  1.703]
    ])
    
    # Reshape for matrix multiplication
    original_shape = ntsc_image.shape
    ntsc_flat = ntsc_image.reshape(-1, 3)
    
    # Apply transformation
    rgb_flat = ntsc_flat @ transform_matrix.T
    
    return rgb_flat.reshape(original_shape)

def blur_down_color(image, levels, filt='binom5'):
    """
    Blur and downsample image (matches MATLAB blurDnClr).
    
    Args:
        image: Input image in NTSC format
        levels: Number of downsampling levels
        filt: Filter type (default 'binom5')
    """
    if filt == 'binom5':
        # Create 2D binomial filter (5x5)
        kernel_1d = binomial_filter(5)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
    else:
        raise ValueError(f"Filter {filt} not implemented")
    
    result = image.copy()
    
    # Apply blur and downsample for each level
    for level in range(levels):
        # Apply separable convolution for each channel
        for c in range(result.shape[2]):
            # Blur
            blurred = cv2.filter2D(result[:,:,c], -1, kernel_2d, borderType=cv2.BORDER_REFLECT)
            result[:,:,c] = blurred
        
        # Downsample by factor of 2
        result = result[::2, ::2, :]
    
    return result

def build_gaussian_down_stack(video_frames, levels):
    """
    Build Gaussian downsampling stack for all frames (matches MATLAB build_GDown_stack).
    
    Args:
        video_frames: Array of video frames in RGB format
        levels: Number of pyramid levels
        
    Returns:
        Stack of downsampled frames
    """
    n_frames = len(video_frames)
    
    # Convert first frame to get dimensions
    first_frame_ntsc = rgb2ntsc(video_frames[0].astype(np.float64) / 255.0)
    first_blurred = blur_down_color(first_frame_ntsc, levels)
    
    # Initialize stack
    stack_shape = (n_frames,) + first_blurred.shape
    gaussian_stack = np.zeros(stack_shape, dtype=np.float64)
    
    # Process each frame
    print("Building Gaussian downsampling stack...")
    for i in tqdm.tqdm(range(n_frames), desc="Spatial filtering"):
        # Convert to NTSC and normalize
        frame_ntsc = rgb2ntsc(video_frames[i].astype(np.float64) / 255.0)
        # Apply Gaussian blur and downsampling
        blurred = blur_down_color(frame_ntsc, levels)
        gaussian_stack[i] = blurred
    
    return gaussian_stack

def ideal_bandpass_filter(input_stack, dim, wl, wh, sampling_rate):
    """
    Apply ideal bandpass filter (matches MATLAB ideal_bandpassing).
    
    Args:
        input_stack: Input stack of frames
        dim: Dimension along which to filter (1 for time axis)
        wl: Lower cutoff frequency
        wh: Higher cutoff frequency  
        sampling_rate: Sampling rate (fps)
    """
    # Shift dimensions to put time axis first (matches MATLAB shiftdim)
    input_shifted = np.moveaxis(input_stack, dim-1, 0)
    dimensions = input_shifted.shape
    
    n = dimensions[0]  # Number of frames
    
    # Create frequency mask
    freq = np.arange(n) / n * sampling_rate
    mask = (freq > wl) & (freq < wh)
    
    # Apply FFT along time axis
    F = np.fft.fft(input_shifted, axis=0)
    
    # Apply frequency mask - broadcast mask to all dimensions
    for i in range(n):
        if not mask[i]:
            F[i] = 0
    
    # Inverse FFT
    filtered = np.real(np.fft.ifft(F, axis=0))
    
    # Shift dimensions back
    filtered = np.moveaxis(filtered, 0, dim-1)
    
    return filtered

def amplify_spatial_gdown_temporal_ideal(video_path, output_path, alpha, level, fl, fh, chrom_attenuation=1.0):
    """
    Eulerian Video Magnification using Gaussian downsampling and ideal temporal filtering.
    Exactly matches MATLAB amplify_spatial_Gdown_temporal_ideal function.
    
    Args:
        video_path: Input video file path
        output_path: Output video file path
        alpha: Amplification factor
        level: Number of pyramid levels
        fl: Lower frequency bound (Hz)
        fh: Higher frequency bound (Hz)
        chrom_attenuation: Chrominance attenuation factor
    """
    print("Eulerian Video Magnification - MATLAB Compatible")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Alpha: {alpha}, Levels: {level}")
    print(f"Frequency band: {fl:.3f} - {fh:.3f} Hz")
    print(f"Chrominance attenuation: {chrom_attenuation}")
    print("-" * 50)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {frame_count} frames, {width}x{height}, {fps:.1f} FPS")
    
    # Load all frames
    frames = []
    print("Loading video frames...")
    for i in tqdm.tqdm(range(frame_count), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    frames = np.array(frames)
    
    # Build Gaussian downsampling stack
    gaussian_stack = build_gaussian_down_stack(frames, level)
    
    # Apply temporal filtering
    print("Applying temporal filtering...")
    filtered_stack = ideal_bandpass_filter(gaussian_stack, 1, fl, fh, fps)
    
    # Apply amplification (matches MATLAB exactly)
    print("Applying amplification...")
    filtered_stack[:,:,:,0] *= alpha  # Y channel
    filtered_stack[:,:,:,1] *= alpha * chrom_attenuation  # I channel  
    filtered_stack[:,:,:,2] *= alpha * chrom_attenuation  # Q channel
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Render output video
    print("Rendering output video...")
    for i in tqdm.tqdm(range(len(frames)), desc="Rendering"):
        # Get original frame in NTSC
        original_rgb = frames[i].astype(np.float64) / 255.0
        original_ntsc = rgb2ntsc(original_rgb)
        
        # Get filtered frame and resize to original dimensions
        filtered_frame = filtered_stack[i]
        filtered_resized = cv2.resize(filtered_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Add filtered to original (in NTSC space)
        combined_ntsc = filtered_resized + original_ntsc
        
        # Convert back to RGB
        combined_rgb = ntsc2rgb(combined_ntsc)
        
        # Clamp values to [0, 1] (matches MATLAB)
        combined_rgb = np.clip(combined_rgb, 0, 1)
        
        # Convert to uint8 and BGR for OpenCV
        output_frame = (combined_rgb * 255).astype(np.uint8)
        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        out.write(output_frame_bgr)
    
    out.release()
    print("Processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Eulerian Video Magnification - MATLAB Compatible')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path')
    parser.add_argument('--alpha', type=float, default=50, help='Amplification factor (default: 50)')
    parser.add_argument('--levels', type=int, default=4, help='Pyramid levels (default: 4)')
    parser.add_argument('--fl', type=float, help='Lower frequency bound (Hz)')
    parser.add_argument('--fh', type=float, help='Higher frequency bound (Hz)')
    parser.add_argument('--chrom-attenuation', type=float, default=1.0, help='Chrominance attenuation (default: 1.0)')
    parser.add_argument('--mode', choices=['pulse', 'respiration'], default='pulse', help='Processing mode')
    
    args = parser.parse_args()
    
    # Set frequency bounds based on mode if not specified
    if args.fl is None or args.fh is None:
        if args.mode == 'pulse':
            args.fl, args.fh = PULSE_BAND_HZ
        else:
            args.fl, args.fh = RESP_BAND_HZ
    
    # Generate output path if not specified
    if args.output is None:
        input_path = Path(args.input_video)
        args.output = str(input_path.parent / f"{input_path.stem}_evm_matlab{input_path.suffix}")
    
    try:
        amplify_spatial_gdown_temporal_ideal(
            args.input_video, 
            args.output, 
            args.alpha, 
            args.levels, 
            args.fl, 
            args.fh, 
            args.chrom_attenuation
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())