# ECE 420 Final Project - Physiological Signal Detection with Visual Magnification

This project implements real-time physiological signal detection from video using computer vision techniques, enhanced with **Eulerian Video Magnification (EVM)** for visual amplification of physiological signals.

## 🎯 Key Features

- **Pulse Detection**: Extract heart rate from facial video using photoplethysmography (PPG)
- **Respiration Detection**: Monitor breathing rate from chest movement analysis
- **🆕 Visual Magnification**: Real-time Eulerian Video Magnification to make physiological signals visible
- **🆕 Fast Startup**: Reduced "warming up" time - detection starts in ~3 seconds
- **Live Camera Demo**: Real-time processing with webcam input and interactive controls
- **Video Processing**: Batch analysis of recorded videos

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the enhanced live camera demo:
```bash
python scripts/live_camera.py
```

3. Or use the interactive demo launcher:
```bash
python demo_live_magnification.py
```

## 🎮 Interactive Controls

During live camera operation:
- **'q'**: Quit the application
- **'s'**: Switch between pulse and respiration detection modes
- **'m'**: Toggle visual magnification on/off

## 🔧 Major Improvements

### ✅ Fixed "Warming Up" Issue
- **Before**: Required 60+ frames and long initialization
- **After**: Starts detecting in ~3 seconds with adaptive parameters
- Reduced minimum frame requirements from 4×FPS to 2×FPS
- Adaptive detrending window based on available data

### ✅ Added Visual Magnification
- **Real-time Eulerian Video Magnification (EVM)**
- Makes subtle physiological changes visible to the naked eye
- Gaussian pyramid decomposition with temporal bandpass filtering
- Adjustable magnification factor (default: 30x)
- Toggle on/off during live operation

### ✅ Enhanced User Experience
- Better error messages and status feedback
- Improved FPS calculation and display
- More responsive parameter updates
- Cleaner visual interface with magnification status

## 📊 Technical Details

### Signal Extraction
- **Pulse**: Green channel analysis from facial ROI (photoplethysmography)
- **Respiration**: Luma (brightness) analysis from chest ROI

### Visual Magnification Pipeline
1. **Gaussian Pyramid**: Multi-scale image decomposition
2. **Temporal Filtering**: Bandpass filtering in frequency domain
3. **Amplification**: Magnify filtered signals by configurable factor
4. **Reconstruction**: Rebuild image with amplified physiological signals

### Signal Processing Pipeline
1. **ROI Detection**: Automatic face/chest region detection
2. **Signal Extraction**: Channel-specific signal extraction
3. **Detrending**: Adaptive moving average detrending
4. **Filtering**: Butterworth bandpass filtering for target frequency ranges
5. **Frequency Analysis**: FFT-based spectrum analysis with peak detection

### Frequency Bands
- **Pulse**: 0.8-1.5 Hz (48-90 BPM) - optimized for quick detection
- **Respiration**: 0.2-0.5 Hz (12-30 breaths/min)

## 🚀 Usage Examples

### Enhanced Live Camera Demo
```bash
# Basic usage with magnification enabled (default)
python scripts/live_camera.py

# Pulse mode with high magnification
python scripts/live_camera.py --mode pulse --mag-factor 35

# Respiration mode
python scripts/live_camera.py --mode respiration

# Disable magnification
python scripts/live_camera.py --magnify false

# Custom parameters with fast updates
python scripts/live_camera.py --window-sec 10 --update-interval 0.2
```

### Video Processing
```bash
# Process video with default settings
python scripts/process_video.py video.mp4

# Specify output directory and mode
python scripts/process_video.py video.mp4 --output results/ --mode respiration

# Custom frequency band
python scripts/process_video.py video.mp4 --band 0.5 2.0
```

## 📁 Project Structure

```
python/
├── physio/                    # Core processing modules
│   ├── config.py             # Configuration parameters
│   ├── filters.py            # Signal filtering functions
│   ├── io.py                # Video I/O utilities
│   ├── pipelines.py         # Processing pipelines
│   ├── roi.py               # Region of interest detection
│   ├── spectrum.py          # Frequency analysis
│   ├── types.py             # Type definitions
│   └── viz.py               # Visualization utilities
├── scripts/                  # Main application scripts
│   ├── live_camera.py       # 🆕 Enhanced live camera with EVM
│   └── process_video.py     # Video processing script
├── demo_live_magnification.py # 🆕 Interactive demo launcher
└── test_physio.py           # Unit tests
```

## ⚙️ Configuration

Key parameters can be adjusted in `physio/config.py` or via command line:

- `PULSE_BAND_HZ`: Frequency range for pulse detection (0.8-1.5 Hz)
- `RESP_BAND_HZ`: Frequency range for respiration detection (0.2-0.5 Hz)
- `LIVE_WINDOW_SEC`: Rolling window size for live processing (15s)
- `--mag-factor`: Visual magnification factor (default: 30)
- `--update-interval`: Rate update frequency (default: 0.2s)

## 🔬 Algorithm Details

### Eulerian Video Magnification
The EVM implementation uses:
- **Spatial Decomposition**: 4-level Gaussian pyramid
- **Temporal Filtering**: Simple bandpass filter using moving averages
- **Magnification**: Linear amplification of filtered signals
- **Reconstruction**: Pyramid reconstruction with clipping to valid pixel ranges

### Adaptive Processing
- **Dynamic Window Sizing**: Adjusts detrending window based on available data
- **Minimum Frame Requirements**: Reduced for faster startup
- **Error Handling**: Graceful degradation with user-friendly messages

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2) >= 4.5.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0 (with fallback implementations)
- Matplotlib >= 3.5.0

## 🧪 Testing

Run the test suite:
```bash
python test_physio.py
```

## 💡 Usage Tips

### For Best Results:
- **Lighting**: Ensure good, even lighting on face/chest
- **Positioning**: Keep face clearly visible for pulse detection
- **Stability**: Minimize camera movement for better magnification
- **Distance**: Position 2-3 feet from camera for optimal ROI detection

### Troubleshooting:
- If magnification appears noisy, reduce `--mag-factor`
- For faster detection, decrease `--update-interval`
- If detection is unstable, increase `--window-sec`

## 🎬 Demo Video

The enhanced live camera demo showcases:
1. **Real-time rate detection** with numerical display
2. **Visual magnification** showing amplified physiological signals
3. **Interactive mode switching** between pulse and respiration
4. **Magnification toggle** to compare original vs. amplified video

## 📝 Notes

- Visual magnification may take 10-15 frames to stabilize
- NumPy 2.x compatibility warnings are normal and don't affect functionality
- Press 'm' during live demo to see the dramatic difference magnification makes
- The system works best with subtle movements - avoid excessive motion

## License

This project is for educational and research purposes.