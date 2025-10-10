#!/usr/bin/env python3
"""
ECE 420 Lab 4 - Pitch Detection Implementation
Author: Generated for Lab 4
Date: 2024

This file implements pitch detection using:
1. Voiced/Unvoiced Detection using energy threshold
2. Autocorrelation-based pitch detection
3. Pitch estimation for voiced frames
4. Visualization of results

Structured for easy conversion to Jupyter Notebook format.
"""

# =============================================================================
# # ECE 420 Lab 4 - Pitch Detection
# 
# This lab implements pitch detection algorithms for audio signals using:
# - Energy-based voiced/unvoiced detection
# - Autocorrelation for pitch period estimation
# - FFT-based efficient autocorrelation computation
# =============================================================================

# =============================================================================
# ## Imports and Constants
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import correlate
import os

# Constants
FRAME_SIZE = 2048
SAMPLE_RATE = 44100  # Default sample rate
MIN_PITCH_FREQ = 80   # Minimum pitch frequency (Hz)
MAX_PITCH_FREQ = 400  # Maximum pitch frequency (Hz)
ENERGY_THRESHOLD = 1e7  # Energy threshold for voiced/unvoiced detection

# =============================================================================
# ## Part 1: Voiced/Unvoiced Detection
# 
# **Question**: How do we decide if a frame is voiced?
# 
# **Answer**: We calculate the energy of the frame and compare it to a threshold.
# If energy > threshold → voiced (1), else unvoiced (0).
# =============================================================================

def calculate_frame_energy(frame):
    """
    Calculate the energy of a frame.
    
    Args:
        frame (np.array): Audio frame
        
    Returns:
        float: Energy of the frame
    """
    return np.sum(frame**2)

def voiced_unvoiced_detection(frame, threshold=ENERGY_THRESHOLD):
    """
    Determine if a frame is voiced or unvoiced based on energy threshold.
    
    Args:
        frame (np.array): Audio frame
        threshold (float): Energy threshold
        
    Returns:
        int: 1 if voiced, 0 if unvoiced
    """
    energy = calculate_frame_energy(frame)
    return 1 if energy > threshold else 0

def ece420ProcessFrame(frame, threshold=ENERGY_THRESHOLD):
    """
    Main processing function for voiced/unvoiced detection.
    This matches the function signature from the prelab.
    
    Args:
        frame (np.array): Audio frame
        threshold (float): Energy threshold
        
    Returns:
        int: 1 if voiced, 0 if unvoiced
    """
    return voiced_unvoiced_detection(frame, threshold)

# =============================================================================
# ## Part 2: Autocorrelation Implementation
# 
# **Question**: Why is autocorrelation useful for pitch detection?
# 
# **Answer**: Autocorrelation finds the pitch period by identifying the lag 
# at which the signal has maximum similarity with itself. For periodic signals 
# (voiced speech), this corresponds to the fundamental period.
# =============================================================================

def autocorrelation_time_domain(signal):
    """
    Compute autocorrelation using time-domain method (O(N^2)).
    
    Args:
        signal (np.array): Input signal
        
    Returns:
        np.array: Autocorrelation sequence
    """
    N = len(signal)
    autocorr = np.zeros(N)
    
    for lag in range(N):
        for n in range(N - lag):
            autocorr[lag] += signal[n] * signal[n + lag]
    
    return autocorr

def autocorrelation_fft(signal):
    """
    Compute autocorrelation using FFT method (O(N log N)).
    This is more efficient for larger signals.
    
    Args:
        signal (np.array): Input signal
        
    Returns:
        np.array: Autocorrelation sequence
    """
    # Zero-pad the signal to avoid circular correlation
    N = len(signal)
    padded_signal = np.concatenate([signal, np.zeros(N)])
    
    # Compute FFT
    fft_signal = np.fft.fft(padded_signal)
    
    # Compute power spectral density
    psd = fft_signal * np.conj(fft_signal)
    
    # Inverse FFT to get autocorrelation
    autocorr = np.real(np.fft.ifft(psd))
    
    # Return only the first N points (positive lags)
    return autocorr[:N]

def find_pitch_period(autocorr, sample_rate, min_freq=MIN_PITCH_FREQ, max_freq=MAX_PITCH_FREQ):
    """
    Find the pitch period from autocorrelation sequence.
    
    Args:
        autocorr (np.array): Autocorrelation sequence
        sample_rate (int): Sample rate of the signal
        min_freq (float): Minimum expected pitch frequency
        max_freq (float): Maximum expected pitch frequency
        
    Returns:
        tuple: (pitch_period_samples, pitch_frequency_hz)
    """
    # Convert frequency limits to sample limits
    max_lag = int(sample_rate / min_freq)
    min_lag = int(sample_rate / max_freq)
    
    # Ensure we don't exceed autocorrelation length
    max_lag = min(max_lag, len(autocorr) - 1)
    min_lag = max(min_lag, 1)
    
    if min_lag >= max_lag:
        return 0, 0
    
    # Find the lag with maximum autocorrelation (excluding lag 0)
    search_range = autocorr[min_lag:max_lag]
    if len(search_range) == 0:
        return 0, 0
        
    max_idx = np.argmax(search_range)
    pitch_period = min_lag + max_idx
    
    # Convert to frequency
    pitch_frequency = sample_rate / pitch_period if pitch_period > 0 else 0
    
    return pitch_period, pitch_frequency

# =============================================================================
# ## Part 3: Complete Pitch Detection Pipeline
# =============================================================================

def pitch_detection_frame(frame, sample_rate, threshold=ENERGY_THRESHOLD, use_fft=True):
    """
    Complete pitch detection for a single frame.
    
    Args:
        frame (np.array): Audio frame
        sample_rate (int): Sample rate
        threshold (float): Energy threshold for voiced/unvoiced detection
        use_fft (bool): Whether to use FFT-based autocorrelation
        
    Returns:
        tuple: (is_voiced, pitch_frequency, pitch_period)
    """
    # Step 1: Voiced/Unvoiced detection
    is_voiced = ece420ProcessFrame(frame, threshold)
    
    if not is_voiced:
        return 0, 0, 0
    
    # Step 2: Autocorrelation-based pitch detection
    if use_fft:
        autocorr = autocorrelation_fft(frame)
    else:
        autocorr = autocorrelation_time_domain(frame)
    
    # Step 3: Find pitch period and frequency
    pitch_period, pitch_frequency = find_pitch_period(autocorr, sample_rate)
    
    return is_voiced, pitch_frequency, pitch_period

def process_audio_file(filename, frame_size=FRAME_SIZE, hop_size=None, threshold=ENERGY_THRESHOLD):
    """
    Process an entire audio file for pitch detection.
    
    Args:
        filename (str): Path to audio file
        frame_size (int): Size of each frame
        hop_size (int): Hop size between frames (default: frame_size)
        threshold (float): Energy threshold
        
    Returns:
        tuple: (sample_rate, voiced_frames, pitch_frequencies, pitch_periods)
    """
    if hop_size is None:
        hop_size = frame_size
    
    try:
        # Read audio file
        sample_rate, data = read(filename)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Convert to float
        data = data.astype(float)
        
        # Calculate number of frames
        num_frames = (len(data) - frame_size) // hop_size + 1
        
        # Initialize result arrays
        voiced_frames = np.zeros(num_frames)
        pitch_frequencies = np.zeros(num_frames)
        pitch_periods = np.zeros(num_frames)
        
        # Process each frame
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            frame = data[start_idx:end_idx]
            
            # Perform pitch detection
            is_voiced, pitch_freq, pitch_period = pitch_detection_frame(
                frame, sample_rate, threshold
            )
            
            voiced_frames[i] = is_voiced
            pitch_frequencies[i] = pitch_freq
            pitch_periods[i] = pitch_period
        
        return sample_rate, voiced_frames, pitch_frequencies, pitch_periods
        
    except Exception as e:
        print(f"Error processing audio file {filename}: {e}")
        return None, None, None, None

# =============================================================================
# ## Part 4: Visualization Functions
# =============================================================================

def plot_voiced_unvoiced(voiced_frames, title="Voiced/Unvoiced Detection"):
    """
    Plot voiced/unvoiced detection results.
    
    Args:
        voiced_frames (np.array): Array of voiced/unvoiced decisions
        title (str): Plot title
    """
    plt.figure(figsize=(12, 4))
    plt.stem(voiced_frames, basefmt=" ")
    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Voiced (1) / Unvoiced (0)")
    plt.ylim(-0.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_pitch_frequencies(pitch_frequencies, voiced_frames, sample_rate, 
                          frame_size=FRAME_SIZE, title="Pitch Frequency Detection"):
    """
    Plot pitch frequency detection results.
    
    Args:
        pitch_frequencies (np.array): Array of detected pitch frequencies
        voiced_frames (np.array): Array of voiced/unvoiced decisions
        sample_rate (int): Sample rate
        frame_size (int): Frame size
        title (str): Plot title
    """
    # Create time axis
    time_per_frame = frame_size / sample_rate
    time_axis = np.arange(len(pitch_frequencies)) * time_per_frame
    
    # Only plot frequencies for voiced frames
    voiced_frequencies = pitch_frequencies.copy()
    voiced_frequencies[voiced_frames == 0] = np.nan
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, voiced_frequencies, 'bo-', markersize=4, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch Frequency (Hz)")
    plt.grid(True, alpha=0.3)
    plt.ylim(MIN_PITCH_FREQ - 20, MAX_PITCH_FREQ + 20)
    plt.show()

def plot_autocorrelation_example(signal, sample_rate, title="Autocorrelation Example"):
    """
    Plot an example of autocorrelation for a signal frame.
    
    Args:
        signal (np.array): Signal frame
        sample_rate (int): Sample rate
        title (str): Plot title
    """
    autocorr = autocorrelation_fft(signal)
    
    # Create lag axis
    lags = np.arange(len(autocorr))
    
    plt.figure(figsize=(12, 8))
    
    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title(f"{title} - Original Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Plot autocorrelation
    plt.subplot(2, 1, 2)
    plt.plot(lags, autocorr)
    plt.title(f"{title} - Autocorrelation")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Autocorrelation")
    plt.grid(True, alpha=0.3)
    
    # Mark potential pitch period
    pitch_period, pitch_freq = find_pitch_period(autocorr, sample_rate)
    if pitch_period > 0:
        plt.axvline(x=pitch_period, color='r', linestyle='--', 
                   label=f'Pitch Period: {pitch_period} samples ({pitch_freq:.1f} Hz)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# ## Part 5: Test and Demo Functions
# =============================================================================

def generate_test_signal(frequency, duration, sample_rate, noise_level=0.1, amplitude=1000):
    """
    Generate a test signal with known pitch for validation.
    
    Args:
        frequency (float): Fundamental frequency
        duration (float): Duration in seconds
        sample_rate (int): Sample rate
        noise_level (float): Noise level (0-1)
        amplitude (float): Signal amplitude to ensure it passes energy threshold
        
    Returns:
        np.array: Generated test signal
    """
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Generate harmonic signal (fundamental + harmonics)
    signal = np.sin(2 * np.pi * frequency * t)  # Fundamental
    signal += 0.5 * np.sin(2 * np.pi * 2 * frequency * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * frequency * t)  # 3rd harmonic
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(len(signal))
        signal += noise
    
    # Scale to desired amplitude
    signal = signal * amplitude
    
    return signal

def test_pitch_detection():
    """
    Test the pitch detection algorithm with known signals.
    """
    print("=" * 60)
    print("Testing Pitch Detection Algorithm")
    print("=" * 60)
    
    # Test parameters
    test_frequencies = [100, 150, 200, 300]
    duration = 0.5  # seconds
    sample_rate = 44100
    
    for freq in test_frequencies:
        print(f"\nTesting with {freq} Hz signal...")
        
        # Generate test signal
        test_signal = generate_test_signal(freq, duration, sample_rate)
        
        # Extract a frame for testing
        frame = test_signal[:FRAME_SIZE]
        
        # Perform pitch detection
        is_voiced, detected_freq, pitch_period = pitch_detection_frame(
            frame, sample_rate
        )
        
        print(f"  Expected frequency: {freq} Hz")
        print(f"  Detected frequency: {detected_freq:.1f} Hz")
        print(f"  Error: {abs(detected_freq - freq):.1f} Hz")
        print(f"  Voiced: {'Yes' if is_voiced else 'No'}")
        
        # Show autocorrelation plot for first test
        if freq == test_frequencies[0]:
            plot_autocorrelation_example(frame, sample_rate, 
                                       f"Autocorrelation for {freq} Hz Signal")

def demo_with_audio_file(filename):
    """
    Demonstrate pitch detection with an audio file.
    
    Args:
        filename (str): Path to audio file
    """
    print(f"\nProcessing audio file: {filename}")
    
    if not os.path.exists(filename):
        print(f"File {filename} not found. Creating a test signal instead.")
        
        # Create a test signal and save it
        test_signal = generate_test_signal(150, 2.0, SAMPLE_RATE)
        test_signal_int = (test_signal * 32767).astype(np.int16)
        write(filename, SAMPLE_RATE, test_signal_int)
        print(f"Created test signal: {filename}")
    
    # Process the audio file
    sample_rate, voiced_frames, pitch_frequencies, pitch_periods = process_audio_file(
        filename, threshold=ENERGY_THRESHOLD
    )
    
    if sample_rate is not None:
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Number of frames: {len(voiced_frames)}")
        print(f"Voiced frames: {np.sum(voiced_frames)} / {len(voiced_frames)}")
        
        # Plot results
        plot_voiced_unvoiced(voiced_frames)
        plot_pitch_frequencies(pitch_frequencies, voiced_frames, sample_rate)
        
        # Print statistics
        voiced_pitches = pitch_frequencies[voiced_frames == 1]
        if len(voiced_pitches) > 0:
            print(f"\nPitch Statistics (voiced frames only):")
            print(f"  Mean pitch: {np.mean(voiced_pitches):.1f} Hz")
            print(f"  Std pitch: {np.std(voiced_pitches):.1f} Hz")
            print(f"  Min pitch: {np.min(voiced_pitches):.1f} Hz")
            print(f"  Max pitch: {np.max(voiced_pitches):.1f} Hz")

# =============================================================================
# ## Main Execution
# =============================================================================

if __name__ == "__main__":
    print("ECE 420 Lab 4 - Pitch Detection")
    print("=" * 40)
    
    # Test with synthetic signals
    test_pitch_detection()
    
    # Test with audio file (if available)
    test_audio_files = [
        "test_vector.wav",
        "sample_audio.wav",
        "test_signal.wav"
    ]
    
    for audio_file in test_audio_files:
        if os.path.exists(audio_file):
            demo_with_audio_file(audio_file)
            break
    else:
        # No audio file found, create and test with synthetic signal
        print("\nNo test audio file found. Creating synthetic test signal...")
        demo_with_audio_file("synthetic_test.wav")
    
    print("\n" + "=" * 60)
    print("Lab 4 Python implementation completed!")
    print("Ready for conversion to Jupyter Notebook.")
    print("=" * 60)