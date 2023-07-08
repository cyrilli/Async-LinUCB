import numpy as np
import random
import copy
import datetime
import sys

sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles')
sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/MovieLen')
from Articles import ArticleManager
from Users import UserManager
from Movies import MovieManager
from util_functions import featureUniform, gaussianFeature

class LocalClient:
	def __init__(self, featureDimension, theta, sigma, K):
		self.d = featureDimension
		self.theta = theta
		self.sigma = sigma
		self.K = K

		# Sufficient statistics stored on the client #
		# latest local sufficient statistics
		self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
		self.b_local = np.zeros(self.d)
		self.arm_selection_local = np.zeros(self.K)

		# aggregated sufficient statistics recently downloaded
		self.A_uploadbuffer = np.zeros((self.d, self.d))
		self.b_uploadbuffer = np.zeros(self.d)
		self.arm_selection_uploadbuffer = np.zeros(self.K)
	
	def matrix_dot(self, a):
		return np.expand_dims(a, axis=1).dot(np.expand_dims(a, axis=0))

	def localUpdate(self, arm_featureVector, a):
		r = (self.theta.dot(arm_featureVector) + np.random.randn()*self.sigma)
		self.A_local += self.matrix_dot(arm_featureVector)
		self.b_local += arm_featureVector * r
		self.arm_selection_local[a] += 1

		self.A_uploadbuffer += self.matrix_dot(arm_featureVector)
		self.b_uploadbuffer += arm_featureVector * r
		self.arm_selection_uploadbuffer[a] += 1
	
	def localUpdate_tabular(self, arm_featureVector, true_reward, a):
		self.A_local += self.matrix_dot(arm_featureVector)
		self.b_local += arm_featureVector * (true_reward + np.random.randn()*self.sigma)
		self.arm_selection_local[a] += 1

		self.A_uploadbuffer += self.matrix_dot(arm_featureVector)
		self.b_uploadbuffer += arm_featureVector * (true_reward + np.random.randn()*self.sigma)
		self.arm_selection_uploadbuffer[a] += 1

class LinGapE_full:
	def __init__(self, dimension, epsilon, delta, NoiseScale, K, dataset, case, gap):
		self.dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.NoiseScale = NoiseScale
		self.K = K
		self.dataset = dataset
		self.case = case

		# for LinGapE computing
		self.sigma = 1.0
		# self.sigma = NoiseScale
		self.reg = 1
		# self.A = np.eye(self.dimension)*self.reg
		# self.b = np.zeros(self.dimension)

		# use the given article dataset
		# if dataset == 0:
		# 	UM = UserManager(self.dimension, 0, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
		# 	User_filename = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles/usersHomo.dat'
		# 	users = UM.loadHomoUsers(User_filename)
		# 	# For the homogeneous case:
		# 	self.theta = users[0].theta

		# 	AM = ArticleManager(self.dimension, self.dimension, argv={'l2_limit': 1}, theta=self.theta)
		# 	Article_filename = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles/ArticlesForHomo_' + str(gap/10) + '_' + str(self.dimension) + '.dat'
		# 	articles = AM.loadArticles(Article_filename)
		# 	self.X = np.zeros((len(articles), self.dimension), dtype=float)
		# 	self.K = len(articles)
		# 	for i in range(self.K):
		# 		self.X[i] = articles[i].featureVector

		# use the movielen data
		if dataset == 0:
			theta_filename = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/MovieLen/150wtheta.dat'
			self.theta = []
			with open(theta_filename, 'r') as f:
				for line in f:
					self.theta.append(float(line))
			self.theta = np.array(self.theta)
			
			MM = MovieManager(self.dimension, self.K, argv={'l2_limit': 1}, theta = self.theta)
			movie_filname = '/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/MovieLen/MovieLen' + str(gap/10) + '_' + str(self.K) + '.dat'
			movies = MM.loadMovies(movie_filname)
			self.X = np.zeros((len(movies), self.dimension), dtype=float)
			for i in range(self.K):
				self.X[i] = movies[i].featureVector


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
			self.dimension = dimension
			self.K = self.dimension
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
			self.dimension = dimension
			self.K = self.dimension
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

	def inilization_pull(self):
		# initialize by pulling each arm once at first
		for i in range(self.K):
			self.A_aggregated += self.matrix_dot(self.X[i])
			if self.case == 'linear':
				r = (self.theta.dot(self.X[i]) + np.random.randn() * self.sigma)
			if self.case == 'tabular':
				r = (self.t_reward[i] + np.random.randn() * self.sigma)
			self.b_aggregated += self.X[i]*r
		
		# Select-direction after initializing
		self.theta_hat = np.linalg.solve(self.A_aggregated, self.b_aggregated)
		self.est_reward = self.X.dot(self.theta_hat)
		self.it = np.argmax(self.est_reward)
		self.jt = np.argmax(self.est_reward - self.est_reward[self.it] +
				np.array([self.confidence_bound(x - self.X[self.it], self.A_aggregated, self.samplecomplexity) for x in self.X]))
		self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A_aggregated, self.samplecomplexity)
		# print(self.est_reward[self.jt])
		# print(self.est_reward[self.it])
		# print(self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A_aggregated, self.samplecomplexity))
		# exit(0)

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

			# assume have 10 users(clients)
			currentclientID = random.randint(1, 10)
			if currentclientID not in self.clients:
				self.clients[currentclientID] = LocalClient(self.dimension, self.theta, self.sigma, self.K)
				self.A_downloadbuffer[currentclientID] = copy.deepcopy(self.A_aggregated)
				self.b_downloadbuffer[currentclientID] = copy.deepcopy(self.b_aggregated)
				self.arm_selection_downloadbuffer[currentclientID] = copy.deepcopy(self.arm_selection_aggregated)

			# select target arm
			a = self.decide_arm(self.X[self.it]- self.X[self.jt], self.A_aggregated)

			# send the selected arm to client, update local and upload buffer
			if self.case == 'linear':
				self.clients[currentclientID].localUpdate(self.X[a], a)
			elif self.case == 'tabular':
				self.clients[currentclientID].localUpdate_tabular(self.X[a], self.t_reward[a], a)
			
			self.samplecomplexity += 1

			if self.samplecomplexity%5000 == 0:
				print("T:", self.samplecomplexity)
				print("arm_select:",self.arm_selection_aggregated)
				print("totalComm: ",self.totalCommCost)
				print("B:", self.B)
				print("E:", self.epsilon)
				print()


			if self.samplecomplexity % 1 == 0:
				self.totalCommCost += 1
				# update server's aggregated
				self.A_aggregated += self.clients[currentclientID].A_uploadbuffer
				self.b_aggregated += self.clients[currentclientID].b_uploadbuffer
				self.arm_selection_aggregated += self.clients[currentclientID].arm_selection_uploadbuffer
				# update server's download buffer for other clients
				for clientID in self.A_downloadbuffer.keys():
					if clientID != currentclientID:
						self.A_downloadbuffer[clientID] += self.clients[currentclientID].A_uploadbuffer
						self.b_downloadbuffer[clientID] += self.clients[currentclientID].b_uploadbuffer
						self.arm_selection_downloadbuffer[clientID] += self.clients[currentclientID].arm_selection_uploadbuffer
				# clear client's upload buffer
				self.clients[currentclientID].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
				self.clients[currentclientID].b_uploadbuffer = np.zeros(self.dimension)
				self.clients[currentclientID].arm_selection_uploadbuffer = np.zeros(self.K)

				# other agents download the update
				for clientID, client in self.clients.items():
					self.totalCommCost += 1
					client.A_local += self.A_downloadbuffer[clientID]
					client.b_local += self.b_downloadbuffer[clientID]
					client.arm_selection_local += self.arm_selection_downloadbuffer[clientID]

					self.A_downloadbuffer[clientID] = np.zeros((self.dimension, self.dimension))
					self.b_downloadbuffer[clientID] = np.zeros(self.dimension)
					self.arm_selection_downloadbuffer[clientID] = np.zeros(self.K)
				
				# Select_direction
				self.theta_hat = np.linalg.solve(self.A_aggregated, self.b_aggregated)
				self.est_reward = self.X.dot(self.theta_hat)
				self.it = np.argmax(self.est_reward)
				self.jt = np.argmax(self.est_reward - 
									np.max(self.est_reward) + 
									np.array([self.confidence_bound(x - self.X[self.it], self.A_aggregated, self.samplecomplexity) for x in self.X]))
				self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.A_aggregated, self.samplecomplexity)
				# self.B = np.max(self.est_reward-np.max(self.est_reward)+np.array([self.confidence_bound(x - self.X[self.it], self.A, self.samplecomplexity) for x in self.X]))
				# print("Iteration %d"%self.samplecomplexity, " Elapsed time", datetime.datetime.now() - startTime)
		best_arm = self.it
		print()
		print("SampleComplexity: ", self.samplecomplexity)
		print("arm selec: ", self.arm_selection_aggregated)
		print("totalComm: ", self.totalCommCost)
		print("B: ", self.B)
		print("E: ", self.epsilon)
		print("Best arm: ", best_arm)
		# exit(0)
		return self.samplecomplexity, self.arm_selection_aggregated, best_arm, self.totalCommCost
		# return self.samplecomplexity, self.arm_selection_aggregated, best_arm, self.samplecomplexity*2

