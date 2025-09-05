import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV data
csv_filename = 'sample_sensor_data.csv'
data = np.genfromtxt(csv_filename, delimiter=',').T
timestamps = (data[0] - data[0, 0]) / 1000 
accel_data = data[1:4]
gyro_data = data[4:-1]

# Find the peaks using range-based maximum detection
def peak_detection(t, sig):
    peaks = []
    thres = 2
    window = 50  # window size for range-based detection
    N = len(sig)
    
    for i in range(window, N-window):
        if sig[i] > thres:
            # Check if this point is the maximum in its local range
            local_range = sig[i-window:i+window+1]
            if sig[i] == np.max(local_range):
                # Avoid duplicate peaks too close together
                if not peaks or abs(t[i] - peaks[-1][0]) > 0.1:  # 0.1 second minimum separation
                    peaks.append((t[i], sig[i]))
    
    return np.array(peaks)

max_peaks = peak_detection(timestamps, accel_data[0])

# Plot
plt.plot(timestamps, accel_data[0])
if len(max_peaks) > 0:
    plt.scatter(max_peaks[:,0], max_peaks[:,1], color = 'red')
plt.axhline(y=2, color='orange', linestyle='--', label='Threshold = 2')
plt.xlabel("Time")
plt.ylabel("Meters per second")
plt.legend()
plt.show()