""" Fitting Functions """
import autograd.numpy as np 
from autograd import grad
from scipy.optimize import least_squares
import matplotlib.pyplot as plt 

class LinearCalibration:
	def __init__(self, n_sensors, n_data):
		self.n_sensors = n_sensors
		self.n_data = n_data
		self.A = []
		self.x = []
		self.z = []
		for _ in range(n_sensors):
			self.A.append(np.ones((self.n_data, 2)))
			self.x.append(np.array([0, 1]))
			self.z.append(np.ones(self.n_data))

	def read_data(self, measured_data, true_data, index):
		self.A[index][:, 1] = measured_data	
		self.z[index] = true_data

	def fit(self, index):
		AtA = np.matmul(self.A[index].transpose(), self.A[index])
		b = np.matmul(self.A[index].transpose(), self.z[index])
		self.x[index] = np.linalg.solve(AtA, b)

	def print(self, index):
		print("Bias: ", self.x[index][0])
		print("Scale Factor: ", self.x[index][1])

	def plot(self, index):
		plt.plot(np.arange(self.n_data), self.z[index], 'o', color='red')
		plt.plot(np.arange(self.n_data), np.matmul(self.A[index], self.x[index]), 'x', color='blue')
		plt.savefig('linear' + str(index) + '.png')

class NonlinearCalibration:
	def __init__(self, n_sensors, n_data):
		self.n_sensors = n_sensors
		self.n_data = n_data
		self.S = []  # (n_sensors, n_sensors) scale factor matrix; assumed to be symmetric 
		self.b = []  # (n_sensors) bias vector
		self.measured_data = np.zeros((self.n_sensors, self.n_data))
		self.true_data = np.zeros((self.n_sensors, self.n_data))
		self.n_off_diagonal = int(0.5 * (self.n_sensors * (self.n_sensors + 1))) - self.n_sensors
		# initial guess; in order of (diagonal, off-diagonals, bias vector); may want to augment diagonal scale factor 
		self.x0 = np.concatenate((np.ones(self.n_sensors), np.zeros(self.n_off_diagonal), np.zeros(self.n_sensors)))  

	def read_data(self, measured_data, true_data):
		self.measured_data = measured_data
		self.true_data = true_data

	def symmetric_matrix(self, coeff):
		S = np.zeros((self.n_sensors, self.n_sensors))
		cnt = 0
		for i in range(self.n_sensors):
			for j in range(i, self.n_sensors):
				S[i, j] = coeff[cnt]
				S[j, i] = S[i, j]
				cnt += 1
		return S

	def residual(self, coeff):
		S = self.symmetric_matrix(coeff[0 : self.n_sensors + self.n_off_diagonal])
		b = coeff[self.n_sensors + self.n_off_diagonal: ]
		error = np.zeros(self.n_sensors * self.n_data)
		for i in range(self.n_data):
			error[self.n_sensors * i : self.n_sensors * i + self.n_sensors] = np.matmul(S, self.measured_data[:, i]) + b - self.true_data[:, i]
		return error

	def norm_residual(self, coeff):
		return 0.5 * np.linalg.squared_norm(self.residual(coeff))

	def jac_residual(self, coeff):
		return grad(self.norm_residual, coeff)

	def fit(self):
		self.sol = least_squares(self.residual, self.x0)  # can change method if needed, or adjust bounds on scale factor diagonals, off-diagonals, or bias 

	def fit_data(self, measured_data):
		S = self.symmetric_matrix(self.sol.x[0 : self.n_sensors + self.n_off_diagonal])
		b = self.sol.x[self.n_sensors + self.n_off_diagonal:]
		calibrated_data = 0 * measured_data
		for i in range(self.n_data):	
			calibrated_data[:, i] = np.matmul(S, measured_data[:, i]) + b
		return calibrated_data

	def print(self):
		print("Scale Factor: ", self.symmetric_matrix(self.sol.x[0 : self.n_sensors + self.n_off_diagonal]))
		print("Bias: ", self.sol.x[self.n_sensors + self.n_off_diagonal: ])

	def plot(self):
		calibrated_data = self.fit_data(self.measured_data)
		for i in range(self.n_sensors):
			plt.figure(i)
			plt.plot(np.arange(self.n_data), self.true_data[i, :], 'o', color='red')
			plt.plot(np.arange(self.n_data), calibrated_data[i, :], 'x', color='blue')
			plt.savefig('nonlinear' + str(i) + '.png')