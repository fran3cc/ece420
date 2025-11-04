"""
Visualization utilities for signals and spectra.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .types import SignalSeries, Spectrum, RateEstimate


def plot_signal(signal: SignalSeries, title: str = "Signal", 
                output_path: str = None, show: bool = True) -> None:
    """
    Plot time-domain signal.
    
    Args:
        signal: Signal to plot
        title: Plot title
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    time_axis = np.arange(len(signal.values)) / signal.fps
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal.values, 'b-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_spectrum(spectrum: Spectrum, rate_estimate: RateEstimate = None,
                 title: str = "Frequency Spectrum", output_path: str = None, 
                 show: bool = True, freq_range: tuple = None) -> None:
    """
    Plot frequency spectrum with optional peak highlighting.
    
    Args:
        spectrum: Spectrum to plot
        rate_estimate: Rate estimate to highlight peak (optional)
        title: Plot title
        output_path: Path to save plot (optional)
        show: Whether to display plot
        freq_range: Frequency range to display (fmin, fmax)
    """
    freqs = spectrum.freqs_hz
    magnitude = spectrum.magnitude
    
    # Apply frequency range filter if specified
    if freq_range:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        magnitude = magnitude[mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude, 'b-', linewidth=1.5)
    
    # Highlight peak if rate estimate provided
    if rate_estimate and rate_estimate.freq_hz > 0:
        peak_freq = rate_estimate.freq_hz
        if freq_range is None or (freq_range[0] <= peak_freq <= freq_range[1]):
            # Find closest frequency bin
            peak_idx = np.argmin(np.abs(freqs - peak_freq))
            plt.plot(freqs[peak_idx], magnitude[peak_idx], 'ro', markersize=8, 
                    label=f'Peak: {peak_freq:.3f} Hz ({rate_estimate.rate:.1f} rate)')
            plt.axvline(peak_freq, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if rate_estimate:
        plt.legend()
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_processing_pipeline(signal: SignalSeries, filtered_signal: np.ndarray,
                           spectrum: Spectrum, rate_estimate: RateEstimate,
                           title_prefix: str = "", output_path: str = None,
                           show: bool = True) -> None:
    """
    Plot complete processing pipeline: original signal, filtered signal, and spectrum.
    
    Args:
        signal: Original signal
        filtered_signal: Filtered signal array
        spectrum: Frequency spectrum
        rate_estimate: Rate estimate
        title_prefix: Prefix for plot titles
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    time_axis = np.arange(len(signal.values)) / signal.fps
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signal
    axes[0].plot(time_axis, signal.values, 'b-', linewidth=1)
    axes[0].set_title(f'{title_prefix}Original Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Filtered signal
    if len(filtered_signal) == len(time_axis):
        axes[1].plot(time_axis, filtered_signal, 'g-', linewidth=1)
    axes[1].set_title(f'{title_prefix}Filtered Signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Spectrum
    axes[2].plot(spectrum.freqs_hz, spectrum.magnitude, 'r-', linewidth=1.5)
    if rate_estimate.freq_hz > 0:
        peak_idx = np.argmin(np.abs(spectrum.freqs_hz - rate_estimate.freq_hz))
        axes[2].plot(spectrum.freqs_hz[peak_idx], spectrum.magnitude[peak_idx], 
                    'ko', markersize=8, label=f'Peak: {rate_estimate.rate:.1f}')
        axes[2].axvline(rate_estimate.freq_hz, color='black', linestyle='--', alpha=0.7)
    axes[2].set_title(f'{title_prefix}Frequency Spectrum')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()