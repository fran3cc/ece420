import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV data
csv_filename = 'sample_sensor_data.csv'
data = np.genfromtxt(csv_filename, delimiter=',').T
timestamps = (data[0] - data[0, 0]) / 1000 
accel_data = data[1:4]
gyro_data = data[4:-1]

# Find the peak
def peak_detection(t, sig):
    peaks = []
    max_val = -np.Inf
    N = len(sig)

    for i in range(N):
        if sig[i] > max_val:
            max_val = sig[i]
            position = t[i]

    peaks.append((position, max_val))
    return np.array(peaks)

max_peaks = peak_detection(timestamps, accel_data[0])

# Plot
plt.plot(timestamps, accel_data[0])
plt.scatter(max_peaks[:,0], max_peaks[:,1], color = 'red')
plt.xlabel("Time")
plt.ylabel("Meters per second")
plt.show()
