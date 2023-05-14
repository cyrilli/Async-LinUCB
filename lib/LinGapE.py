import numpy as np
import copy
import datetime
import sys
sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles')
from Articles import ArticleManager
from Users import UserManager
from util_functions import featureUniform, gaussianFeature

class LinGapE:
	def __init__(self, dimension, epsilon, delta, NoiseScale, dataset, case, gap):
		self.dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.NoiseScale = NoiseScale
		self.dataset = dataset
		self.case = case

		# for LinGapE computing
		self.sigma = 1.0
		self.reg = 1
		self.A = np.eye(self.dimension)*self.reg
		self.b = np.zeros(self.dimension)
		
		# records
		self.samplecomplexity = 0
		self.totalCommCost = 0

		# use the generated Articles dataset
		if dataset == 0:
			UM = UserManager(self.dimension, 0, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
			User_filename = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles/usersHomo.dat'
			users = UM.loadHomoUsers(User_filename)
			# For the homogeneous case:
			self.theta = users[0].theta

			AM = ArticleManager(self.dimension, 5, argv={'l2_limit': 1}, theta=self.theta)
			Article_filename = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles/ArticlesForHomo_' + str(gap/10) + '_' + str(5) + '.dat'
			articles = AM.loadArticles(Article_filename)
			self.X = np.zeros((len(articles), self.dimension), dtype=float)
			self.K = len(articles)
			for i in range(self.K):
				self.X[i] = articles[i].featureVector
			# self.theta = np.zeros(self.dimension, dtype=float)

		# syn dataset setting 1(corresponding to the LinGapE paper exp 7.1.1)
		if dataset == 1 and case == 'linear':
			self.X = np.eye(self.dimension, dtype = float)
			tmp = np.zeros(self.dimension, dtype=float)
			tmp[0] = np.cos(0.01)
			tmp[1] = np.sin(0.01)
			self.X = np.r_[self.X, np.expand_dims(tmp, axis=0)]
			self.theta = np.zeros((self.dimension,1), dtype=float)
			self.theta[0] = 2
			self.K = len(self.X)


		# syn dataset setting 2(corresponding to the LinGapE paper exp 7.1.2)
		if dataset == 2 and case == 'linear':
			self.K = 5
			self.dimension = 5
			self.X = np.eye(self.dimension, dtype=float)
			self.theta = np.zeros(self.dimension, dtype = float)
			# hyperparameters to control the true reward gap:
			# true reward gap: delta_setting
			# 0.11: 0.1, 0.206: 0.175, 0.305:0.245, 0.406:0.31, 0.503:0.368
			if gap == 1:
				delta_ = 0.1
			elif gap == 2:
				delta_ = 0.175
			elif gap == 3:
				delta_ = 0.245
			elif gap == 4:
				delta_ = 0.31
			elif gap == 5:
				delta_ = 0.368 
			self.theta[0] = delta_ 
			self.X[0] += self.theta

		# syn dataset setting of tabular case
		if dataset == 2 and case == 'tabular':
			self.K = 5
			self.dimension = 5
			self.X = np.eye(self.dimension, dtype=float)
			# self.theta = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
			# delta_ = np.zeros((self.dimension,1), dtype = float)
			self.t_reward = [0.1]*self.K
			if gap == 1:
				delta_ = 0.1
			elif gap == 2:
				delta_ = 0.2
			elif gap == 3:
				delta_ = 0.3
			elif gap == 4:
				delta_ = 0.4
			elif gap == 5:
				delta_ = 0.5
			self.t_reward[0] += delta_
			# print(self.t_reward)
			# exit(0)
			

		# print(self.X)
		# exit(0)

		self.arm_selection = np.ones(self.K)
	
	def getReward(self, arm_vector):
		return self.theta.dot(arm_vector)


	def compute_true_gap(self):
		if self.case == 'linear':
			# compute the true gap of dataset
			reward = np.zeros(self.K,dtype=float)
			for i in range(self.K):
				reward[i] = self.getReward(self.X[i])
			reward = np.sort(reward)
			gap = reward[self.K-1] - reward[self.K-2]
			print("Expected reward gap: ",gap)
		elif self.case == "tabular":
			gap = self.t_reward[0] - self.t_reward[1]
			print("Expected reward gap: ",gap)


	def inilization_pull(self):
		# initialize by pulling each arm once at first
		for i in range(self.K):
			self.A += self.matrix_dot(self.X[i])
			if self.case == 'linear':
				r = (self.theta.dot(self.X[i]) + np.random.randn() * self.sigma)
			if self.case == 'tabular':
				r = (self.t_reward[i] + np.random.randn() * self.sigma)
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

	def run(self):

		# compute the true gap first
		self.compute_true_gap()

		# initialize by pulling each arm once at first
		self.inilization_pull()

		# startTime = datetime.datetime.now()
		while(self.B > self.epsilon):
			# select target arm
			a = self.decide_arm(self.X[self.it]- self.X[self.jt], self.A)

			# Update At and bt--
			self.A += self.matrix_dot(self.X[a])
			if self.case == 'linear':
				self.b += self.X[a] * (self.theta.dot(self.X[a]) + np.random.randn()*self.sigma)
			elif self.case == 'tabular':
				self.b += self.X[a] * (self.t_reward[a] + np.random.randn()*self.sigma)

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

		best_arm = self.it
		return self.samplecomplexity, self.arm_selection, best_arm, 0

