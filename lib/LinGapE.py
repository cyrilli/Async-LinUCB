import numpy as np
import copy
import datetime
import sys

class LinGapE:
	def __init__(self, dimension, epsilon, delta, NoiseScale, articles, dataset):
		self.dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.NoiseScale = NoiseScale
		self.dataset = dataset

		# for LinGapE computing
		self.sigma = 1.0
		self.reg = 1
		self.A = np.eye(self.dimension)*self.reg
		self.b = np.zeros(self.dimension)
		
		# records
		self.samplecomplexity = len(articles)
		self.totalCommCost = 0

		# use the given article dataset
		if dataset == 0:
			self.X = np.zeros((len(articles), self.dimension), dtype=float)
			self.K = len(articles)
			for i in range(self.K):
				self.X[i] = articles[i].featureVector
			self.theta = np.zeros(self.dimension, dtype=float)
			# theta of this dataset should be in user variable which is not completed 
			# try use dataset 1 or dataset 2

		# syn dataset setting 1(corresponding to the LinGapE paper exp 7.1.1)
		if dataset == 1:
			self.X = np.eye(self.dimension, dtype = float)
			tmp = np.zeros(self.dimension, dtype=float)
			tmp[0] = np.cos(0.01)
			tmp[1] = np.sin(0.01)
			self.X = np.r_[self.X, np.expand_dims(tmp, axis=0)]
			self.theta = np.zeros((self.dimension,1), dtype=float)
			self.theta[0] = 2
			self.K = len(self.X)


		# syn dataset setting 2(corresponding to the LinGapE paper exp 7.1.2)
		if dataset == 2:
			self.K = 5
			self.dimension = 5
			self.X = np.ones((self.dimension, self.dimension), dtype=float)
			self.theta = np.zeros((self.dimension,1), dtype = float)
			self.theta[0] = 0.3 # hyperparameters
			self.X -= self.theta

		# print(self.X)
		# exit(0)

		self.arm_selection = np.ones(self.K)
	
	def getReward(self, arm_vector):
		return arm_vector.T.dot(self.theta)


	def compute_true_gap(self):
		# compute the true gap of dataset
		reward = np.zeros(self.K,dtype=float)
		for i in range(self.K):
			reward[i] = self.getReward(self.X[i])
		reward = np.sort(reward)
		if self.dataset == 1:
			gap = reward[self.K-1] - reward[self.K-2]
		if self.dataset == 2:
			gap = reward[1] - reward[0]
		print("Expected reward gap: ",gap)


	def inilization_pull(self, articles):
		# initialize by pulling each arm once at first
		for i in range(self.K):
			self.A += self.matrix_dot(self.X[i])
			r = (self.theta.T.dot(self.X[i]) + np.random.randn() * self.sigma)
			self.b += self.X[i]*r
		
		# Select-direction after initializing
		self.theta_hat = np.linalg.solve(self.A, self.b)
		self.est_reward = self.X.dot(self.theta_hat)
		self.it = np.argmax(self.est_reward)
		self.jt = np.argmax(self.est_reward - self.est_reward[self.it] +
				np.array([self.confidence_bound(x - self.X[self.it], self.A, self.samplecomplexity) for x in self.X]))
		self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A, self.samplecomplexity)

	def matrix_dot(self, a):
		return np.expand_dims(a, axis=1).dot(np.expand_dims(a, axis=0))


	def confidence_bound(self, x, A, t):
		L = 1
		tmp = np.sqrt(x.dot(np.linalg.inv(A)).dot(x))
		return tmp * (self.sigma * np.sqrt(self.dimension * 
				     np.log(self.K * self.K * (1 + t * L * L) / self.reg / self.delta)) + 
					 np.sqrt(self.reg) * 2)

	def decide_arm(self, y,A):
		# select the arm 
		tmp = [y.dot(np.linalg.inv(A + self.matrix_dot(x))).dot(y) for x in (self.X)]
		return np.argmin(tmp)

	def run(self, articles):

		# compute the true gap first
		self.compute_true_gap()

		# initialize by pulling each arm once at first
		self.inilization_pull(articles)

		# startTime = datetime.datetime.now()
		while(self.B > self.epsilon):
			# select target arm
			a = self.decide_arm(self.X[self.it]- self.X[self.jt], self.A)

			# Update At and bt--
			self.A += self.matrix_dot(self.X[a])
			self.b += self.X[a] * (self.theta.T.dot(self.X[a]) + np.random.randn()*self.sigma)
			
			self.arm_selection[a] += 1
			self.samplecomplexity += 1

			if self.samplecomplexity%5000 == 0:
				print("T:", self.samplecomplexity)
				print("arm_select:",self.arm_selection)
				print("B:", self.B)
				print("E:", self.epsilon)
				print()

			
			# Select_direction
			self.theta_hat = np.linalg.solve(self.A, self.b)
			self.est_reward = self.X.dot(self.theta_hat)
			self.it = np.argmax(self.est_reward)
			self.jt = np.argmax(self.est_reward - 
								np.max(self.est_reward) + 
								np.array([self.confidence_bound(x - self.X[self.it], self.A, self.samplecomplexity) for x in self.X]))
			self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A, self.samplecomplexity)
			# self.B = np.max(self.est_reward-np.max(self.est_reward)+np.array([self.confidence_bound(x - self.X[self.it], self.A, self.samplecomplexity) for x in self.X]))
			# print("Iteration %d"%self.samplecomplexity, " Elapsed time", datetime.datetime.now() - startTime)

		return self.samplecomplexity

