import numpy as np
import random
import copy
import datetime
import sys

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles')
from Articles import ArticleManager
from Users import UserManager
from util_functions import featureUniform, gaussianFeature

class LocalClient:
	def __init__(self, featureDimension, theta, sigma, X, delta, reg, t_reward):
		self.d = featureDimension
		self.theta = theta
		self.sigma = sigma
		self.X = X
		self.K = len(self.X)
		self.delta = delta
		self.reg = reg
		self.true_reward = t_reward

		# Sufficient statistics stored on the client #
		# latest local sufficient statistics
		self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
		self.b_local = np.zeros(self.d)
		self.arm_selection_local = np.zeros(self.d)

		# aggregated sufficient statistics recently downloaded
		self.A_uploadbuffer = np.zeros((self.d, self.d))
		self.b_uploadbuffer = np.zeros(self.d)
		self.arm_selection_uploadbuffer = np.zeros(self.d)
	
	def matrix_dot(self, a):
		return np.expand_dims(a, -1).dot(np.expand_dims(a, axis=0))
	
	def confidence_bound(self, x, A):
		det = np.linalg.det(A)
		tmp = np.sqrt(x.dot(np.linalg.inv(A)).dot(x))
		return tmp * (self.sigma * np.sqrt(self.d * 
				     np.log(self.K * self.K * np.sqrt(det) / self.delta)) + 
					 np.sqrt(self.reg) * 2)

	def decide_arm(self, server_ratio):
		self.theta_hat = np.linalg.solve(self.A_local, self.b_local)
		self.est_reward = self.X.dot(self.theta_hat)
		self.it = np.argmax(self.est_reward)
		self.jt = np.argmax(self.est_reward - self.est_reward[self.it] +
				np.array([self.confidence_bound(x - self.X[self.it], self.A_local) for x in self.X]))

		ratio = server_ratio[(self.it, self.jt)]
		a = np.argmin([self.arm_selection_local[i]/(ratio[i] + 1.0e-10) for i in range(self.K)])
		self.localUpdate_tabular(self.X[a], a)

		# a = self.decide_arm(self.ratio[(self.it, self.jt)])

	def localUpdate(self, arm_featureVector, a):
		self.A_local += self.matrix_dot(arm_featureVector)
		self.b_local += arm_featureVector * (self.theta.dot(arm_featureVector) + np.random.randn()*self.sigma)
		self.arm_selection_local[a] += 1

		self.A_uploadbuffer += self.matrix_dot(arm_featureVector)
		self.b_uploadbuffer += arm_featureVector * (self.theta.dot(arm_featureVector) + np.random.randn()*self.sigma)
		self.arm_selection_uploadbuffer[a] += 1
	
	def localUpdate_tabular(self, arm_featureVector, a):
		self.A_local += self.matrix_dot(arm_featureVector)
		self.b_local += arm_featureVector * (self.true_reward[a] + np.random.randn()*self.sigma)
		self.arm_selection_local[a] += 1

		self.A_uploadbuffer += self.matrix_dot(arm_featureVector)
		self.b_uploadbuffer += arm_featureVector * (self.true_reward[a] + np.random.randn()*self.sigma)
		self.arm_selection_uploadbuffer[a] += 1

class UGapE_mult:
	def __init__(self, dimension, epsilon, delta, NoiseScale,  dataset, case, gap):
		self.dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.NoiseScale = NoiseScale
		self.dataset = dataset
		self.case = 'tabular'

		# for LinGapE computing
		self.sigma = 1.0
		self.reg = 1
		# self.A = np.eye(self.dimension)*self.reg
		# self.b = np.zeros(self.dimension)

		# use the given article dataset
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
			self.theta = np.zeros(self.dimension, dtype = float)
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

			# records
		self.samplecomplexity = len(self.X)
		self.totalCommCost = 0

		# client part
		self.clients = {}
		# aggregated sufficient statistics of all clients
		self.A_aggregated = np.eye(self.dimension)*self.reg
		self.b_aggregated = np.zeros(self.dimension)
		self.arm_selection_aggregated = np.ones(self.K) # for inilization, all arm got pulled once
		# aggregated sufficient statistics that haven't been sent to each client
		self.A_downloadbuffer = {}
		self.b_downloadbuffer = {}
		self.arm_selection_downloadbuffer = {}

		# print(self.X)
		# exit(0)
	
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
	
	def get_optimal(self, y):
		names = [str(i) for i in range(self.K)]
		prob = LpProblem("The Whiskas Problem", LpMinimize)
		w = LpVariable.dicts("w", names)
		abs_w = LpVariable.dicts("abs_w", names, lowBound=0)
		prob += lpSum([abs_w[i] for i in names])
		for j in range(self.dimension):
			prob += (lpSum([self.X[int(i), j] * w[i] for i in names]) == y[j])
		for i in names:
			prob += (abs_w[i] >= w[i])
			prob += (abs_w[i] >= -w[i])
		prob.solve(PULP_CBC_CMD(msg=0))
		ratio = np.array([abs_w[i].value() for i in names])
		return ratio

	def inilization_pull(self):
		# initialize by pulling each arm once at first
		for i in range(self.K):
			self.A_aggregated += self.matrix_dot(self.X[i])
			r = (self.t_reward[i] + np.random.randn() * self.sigma)
			self.b_aggregated += self.X[i]*r
		
		for currentclientID in range(10):
			self.clients[currentclientID] = LocalClient(self.dimension, self.theta, self.sigma, self.X, self.delta, self.reg, self.t_reward)
			self.clients[currentclientID].A_local = copy.deepcopy(self.A_aggregated)
			self.clients[currentclientID].b_local = copy.deepcopy(self.b_aggregated)
			self.clients[currentclientID].arm_selection_local = copy.deepcopy(self.arm_selection_aggregated)

		# Select-direction after initializing
		self.theta_hat = np.linalg.solve(self.A_aggregated, self.b_aggregated)
		self.est_reward = self.X.dot(self.theta_hat)
		self.it = np.argmax(self.est_reward)
		self.jt = np.argmax(self.est_reward - self.est_reward[self.it] +
				np.array([self.confidence_bound(x - self.X[self.it], self.A_aggregated) for x in self.X]))
		self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A_aggregated)

	def matrix_dot(self, a):
		return np.expand_dims(a, -1).dot(np.expand_dims(a, axis=0))

	def confidence_bound(self, x, A):
		det = np.linalg.det(A)
		tmp = np.sqrt(x.dot(np.linalg.inv(A)).dot(x))
		return tmp * (self.sigma * np.sqrt(self.dimension * 
				     np.log(self.K * self.K * np.sqrt(det) / self.delta)) + 
					 np.sqrt(self.reg) * 2)

	def decide_arm(self, ratio):
		# select the arm 
		return np.argmin([self.arm_selection_aggregated[i]/(ratio[i] + 1.0e-10) for i in range(self.K)])

	def run(self):

		# compute the true gap first
		self.compute_true_gap()

		# initialize by pulling each arm once at first
		self.inilization_pull()

		self.ratio = dict()
		for i in range(self.K):
			for j in range(self.K):
				self.ratio[(i, j)] = self.get_optimal(self.X[i] - self.X[j])

		# startTime = datetime.datetime.now()
		while(self.B > self.epsilon):

			# assume have 10 users(clients)
			currentclientID = random.randint(0, 9)
			self.clients[currentclientID].decide_arm(self.ratio)
			
			# select target arm
			# a = self.decide_arm(self.ratio[(self.it, self.jt)])

			# send the selected arm to client, update local and upload buffer
			# self.clients[currentclientID].localUpdate_tabular(self.X[a], self.t_reward[a], a)
			
			self.samplecomplexity += 1

			if self.samplecomplexity % 2000 == 0:
				self.totalCommCost += 2
				# update server's aggregated
				self.A_aggregated += self.clients[currentclientID].A_uploadbuffer
				self.b_aggregated += self.clients[currentclientID].b_uploadbuffer
				self.arm_selection_aggregated += self.clients[currentclientID].arm_selection_uploadbuffer
				
				# Select_direction
				self.theta_hat = np.linalg.solve(self.A_aggregated, self.b_aggregated)
				self.est_reward = self.X.dot(self.theta_hat)
				self.it = np.argmax(self.est_reward)
				self.jt = np.argmax(self.est_reward - 
									np.max(self.est_reward) + 
									np.array([self.confidence_bound(x - self.X[self.it], self.A_aggregated) for x in self.X]))
				self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A_aggregated)
				# self.B = np.max(self.est_reward-np.max(self.est_reward)+np.array([self.confidence_bound(x - self.X[self.it], self.A, self.samplecomplexity) for x in self.X]))
				# print("Iteration %d"%self.samplecomplexity, " Elapsed time", datetime.datetime.now() - startTime)
				
				
				self.clients[currentclientID].A_local = copy.deepcopy(self.A_aggregated) 
				self.clients[currentclientID].b_local = copy.deepcopy(self.b_aggregated) 
				self.clients[currentclientID].arm_selection_local = copy.deepcopy(self.arm_selection_aggregated)


				# clear client's upload buffer
				self.clients[currentclientID].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
				self.clients[currentclientID].b_uploadbuffer = np.zeros(self.dimension)
				self.clients[currentclientID].arm_selection_uploadbuffer = np.zeros(self.dimension)
				
				if self.samplecomplexity%5000 == 0:
					print("T:", self.samplecomplexity)
					print("arm_select:",self.arm_selection_aggregated)
					print("totalComm: ",self.totalCommCost)
					print("B:", self.B)
					print("E:", self.epsilon)
					print()

		best_arm = self.it
		return self.samplecomplexity, self.arm_selection_aggregated, best_arm, self.totalCommCost
		# return self.samplecomplexity, self.arm_selection_aggregated, best_arm, self.samplecomplexity*2

