#!/usr/bin/env python3
"""
Demo script to showcase the enhanced live camera with visual magnification.
This script demonstrates both rate detection and visual magnification features.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def main():
    print("=" * 60)
    print("ECE 420 Final Project - Live Physiological Signal Detection")
    print("with Real-time Visual Magnification")
    print("=" * 60)
    print()
    
    print("🎯 FEATURES:")
    print("  ✅ Real-time pulse detection (face ROI)")
    print("  ✅ Real-time respiration detection (chest ROI)")
    print("  ✅ Eulerian Video Magnification (EVM)")
    print("  ✅ Visual amplification of physiological signals")
    print("  ✅ Faster startup (reduced 'warming up' time)")
    print("  ✅ Interactive mode switching")
    print()
    
    print("🎮 CONTROLS:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to switch between pulse/respiration modes")
    print("  • Press 'm' to toggle visual magnification on/off")
    print()
    
    print("🔧 KEY IMPROVEMENTS:")
    print("  • Fixed 'warming up' issue - now starts detecting in ~3 seconds")
    print("  • Added real-time Eulerian Video Magnification")
    print("  • Visual amplification makes pulse/breathing visible to the eye")
    print("  • Adaptive processing parameters for faster response")
    print("  • Better error handling and user feedback")
    print()
    
    print("📊 TECHNICAL DETAILS:")
    print("  • Pulse detection: Green channel analysis on face ROI")
    print("  • Respiration: Luma analysis on chest ROI")
    print("  • EVM: Gaussian pyramid + temporal bandpass filtering")
    print("  • Magnification factor: 25-30x (adjustable)")
    print("  • Frequency bands: 0.8-1.5 Hz (pulse), 0.2-0.5 Hz (respiration)")
    print()
    
    print("🚀 USAGE EXAMPLES:")
    print("  Basic usage:")
    print("    python scripts/live_camera.py")
    print()
    print("  Pulse mode with high magnification:")
    print("    python scripts/live_camera.py --mode pulse --mag-factor 35")
    print()
    print("  Respiration mode:")
    print("    python scripts/live_camera.py --mode respiration")
    print()
    print("  Custom frequency band:")
    print("    python scripts/live_camera.py --band 0.5 2.0")
    print()
    
    response = input("Would you like to start the live demo now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🎬 Starting live demo...")
        print("Make sure your camera is connected and you're in good lighting!")
        print("Position yourself so your face is visible for pulse detection.")
        print()
        
        # Import and run the live camera script
        try:
            from live_camera import main as run_live_camera
            return run_live_camera()
        except ImportError:
            print("❌ Error: Could not import live_camera module")
            print("Please make sure you're running this from the correct directory.")
            return 1
        except Exception as e:
            print(f"❌ Error starting live demo: {e}")
            return 1
    else:
        print("\n👋 Demo cancelled. You can run the live camera script manually:")
        print("   python scripts/live_camera.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())