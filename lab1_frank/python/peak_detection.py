import numpy as np
import matplotlib.pyplot as plt
import os

csv_filename = 'sample_sensor_data.csv'
data = np.genfromtxt(csv_filename, delimiter=',').T
timestamps = (data[0] - data[0, 0]) / 1000

accel_data = data[1:4]
gyro_data = data[4:-1]

plt.plot(timestamps, accel_data[0])
plt.show()