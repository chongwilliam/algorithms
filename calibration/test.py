""" Test Script """
from fit import LinearCalibration, NonlinearCalibration
import numpy as np
import pdb

if __name__ == '__main__':
	### Linear Case ###
	# # Setup class
	# n_sensors = 3
	# n_data = 1000
	# cal = LinearCalibration(n_sensors, n_data)
	
	# # Construct test samples for a SINGLE sensor
	# noise_sf = 1 * 1e-4
	# tau_actual = np.random.uniform(low=-1, high=1, size=(n_data,))
	# x = np.random.rand(2)
	# b, m = x[0], x[1]
	# tau_measured = (tau_actual - b) * (1 / m) 
	# tau_measured += noise_sf * np.random.uniform(low=-1, high=1, size=(n_data,))

	# # Calibrate sensor 0
	# sensor_id = 0
	# cal.read_data(tau_measured, tau_actual, sensor_id)
	# cal.fit(sensor_id)
	# # cal.print(sensor_id)
	# cal.plot(sensor_id)

	# # Print ground truth parameters
	# print("True scale factor: ", m)
	# print("Fit scale factor: ", cal.x[sensor_id][1])
	# print("True bias: ", b)
	# print("Fit bias: ", cal.x[sensor_id][0])

	### Nonlinear Case ###
	# Setup class
	n_sensors = 3
	n_data = 10
	cal = NonlinearCalibration(n_sensors, n_data)
	
	# Construct test samples for all sensors
	tau_actual = np.zeros((n_sensors, n_data))
	tau_measured = np.zeros((n_sensors, n_data))
	noise_sf = 1 * 1e-2
	S_noise = noise_sf * np.random.uniform(low=-1, high=1, size=(n_sensors, n_sensors))
	S = np.diag(np.ones(n_sensors)) + np.matmul(S_noise.transpose(), S_noise)
	b = noise_sf * np.random.uniform(low=-1, high=1, size=(n_sensors))	
	for i in range(n_data):
		tau_actual[:, i] = np.random.uniform(low=-1, high=1, size=(n_sensors,))
		tau_measured[:, i] = np.matmul(np.linalg.inv(S), tau_actual[:, i] - b)

	# Calibrate sensors
	cal.read_data(tau_measured, tau_actual)
	cal.fit()
	cal.plot()

	# Print ground truth parameters
	print("True scale factor: ", S)
	print("True bias: ", b)
	print("Fit: ")
	cal.print()

	# ### Actual Case ###
	# # Setup class
	# n_sensors = 4
	# n_data = 14
	# cal = NonlinearCalibration(n_sensors, n_data)
	# cal_data = np.loadtxt('torques.txt')

	# # Calibrate sensors
	# cal.read_data(tau_measured, tau_actual)
	# cal.fit()
	# cal.plot()

	# # Print ground truth parameters
	# print("Fit: ")
	# cal.print()