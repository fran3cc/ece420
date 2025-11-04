"""
Process video file for pulse or respiration detection.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from physio.io import read_video_frames, save_signal_csv, save_spectrum_csv, save_results_json
from physio.roi import FaceROI, ChestROI
from physio.pipelines import run_pulse, run_respiration
from physio.viz import plot_processing_pipeline
from physio.config import PULSE_BAND_HZ, RESP_BAND_HZ


def parse_args():
    parser = argparse.ArgumentParser(description="Process video for physiological rate detection")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--mode", choices=["pulse", "respiration"], default="pulse",
                       help="Detection mode")
    parser.add_argument("--output-dir", default="outputs", 
                       help="Output directory for results")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--band", type=float, nargs=2, default=None,
                       help="Custom frequency band (fmin fmax)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--save-data", action="store_true",
                       help="Save signal and spectrum data to CSV")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine processing parameters
    if args.mode == "pulse":
        roi_selector = FaceROI()
        band = args.band or PULSE_BAND_HZ
        pipeline_func = run_pulse
        rate_unit = "BPM"
    else:  # respiration
        roi_selector = ChestROI()
        band = args.band or RESP_BAND_HZ
        pipeline_func = run_respiration
        rate_unit = "breaths/min"
    
    print(f"Processing video: {args.video_path}")
    print(f"Mode: {args.mode}")
    print(f"Frequency band: {band[0]:.2f} - {band[1]:.2f} Hz")
    
    try:
        # Read video
        print("Reading video frames...")
        batch = read_video_frames(args.video_path, max_frames=args.max_frames)
        print(f"Loaded {len(batch.frames_bgr)} frames at {batch.fps:.1f} FPS")
        
        # Process
        print("Processing...")
        rate_estimate, spectrum, signal, diagnostics = pipeline_func(batch, roi_selector, band)
        
        # Print results
        print(f"\nResults:")
        print(f"Estimated {args.mode} rate: {rate_estimate.rate:.2f} {rate_unit}")
        print(f"Peak frequency: {rate_estimate.freq_hz:.3f} Hz")
        print(f"ROI coordinates: {diagnostics.roi}")
        
        # Generate base filename
        video_name = Path(args.video_path).stem
        base_name = f"{video_name}_{args.mode}"
        
        # Save results
        results_path = output_dir / f"{base_name}_results.json"
        metadata = {
            "video_path": str(args.video_path),
            "mode": args.mode,
            "frequency_band_hz": band,
            "fps": batch.fps,
            "num_frames": len(batch.frames_bgr),
            "roi": diagnostics.roi
        }
        save_results_json(rate_estimate, str(results_path), metadata)
        print(f"Results saved to: {results_path}")
        
        # Save data if requested
        if args.save_data:
            signal_path = output_dir / f"{base_name}_signal.csv"
            spectrum_path = output_dir / f"{base_name}_spectrum.csv"
            save_signal_csv(signal, str(signal_path))
            save_spectrum_csv(spectrum, str(spectrum_path))
            print(f"Signal data saved to: {signal_path}")
            print(f"Spectrum data saved to: {spectrum_path}")
        
        # Generate plots if requested
        if not args.no_plots:
            plot_path = output_dir / f"{base_name}_analysis.png"
            
            # Create filtered signal for visualization
            from physio.filters import moving_average_detrend, butter_bandpass
            from physio.config import DETREND_WINDOW_SEC, DETREND_WINDOW_RESP_SEC
            
            window_sec = DETREND_WINDOW_SEC if args.mode == "pulse" else DETREND_WINDOW_RESP_SEC
            detrended = moving_average_detrend(signal.values, signal.fps, window_sec)
            filtered = butter_bandpass(detrended, signal.fps, band[0], band[1])
            
            plot_processing_pipeline(
                signal, filtered, spectrum, rate_estimate,
                title_prefix=f"{args.mode.title()} Detection - ",
                output_path=str(plot_path),
                show=False
            )
            print(f"Analysis plot saved to: {plot_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())