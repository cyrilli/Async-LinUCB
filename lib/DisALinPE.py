import numpy as np
import copy
import random

import sys
sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB/Dataset/SimArticles')
from Articles import ArticleManager
from Users import UserManager
from util_functions import featureUniform, gaussianFeature

class LocalClient:
	def __init__(self, K, featureDimension, theta, noise, reg, X, gamma1, gamma2, n_clients):
		self.K = K
		self.d = featureDimension
		self.theta = theta
		self.noise = noise
		self.reg = reg
		self.X = X
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.n_clients = n_clients


		self.V_local = np.zeros((self.d, self.d))
		self.b_local = np.zeros(self.K)
		self.arm_selection_local = np.zeros(self.K)

		self.V_uploadbuffer = np.zeros((self.d, self.K))
		self.b_uploadbuffer = np.zeros(self.d)
		self.arm_selection_uploadbuffer = np.zeros(self.K)

	def matrix_dot(self, a):
		return np.expand_dims(a, axis=-1).dot(np.expand_dims(a, axis=0))

	def confidence_bound(self, x, V):
		delta = 0.05
		tmp = np.sqrt(x.dot(np.linalg.inv(V)).dot(x))
		C1 = (np.sqrt(2*self.gamma1)*self.n_clients) + (np.sqrt(1+self.gamma1*self.n_clients))
		C2 = self.noise*(np.sqrt(self.d*np.log(2/delta*(1+(1+self.gamma2*self.n_clients)*np.sum(self.arm_selection_local)/(min(self.gamma1, 1)*self.reg)))))
		return tmp*(np.sqrt(self.reg) + C1*C2)

	def localupdate(self, r, a):
		self.V_local += self.matrix_dot(self.X[a])
		self.b_local += self.X[a]*r
		self.arm_selection_local[a] += 1

		self.V_uploadbuffer += self.matrix_dot(self.X[a])
		self.b_uploadbuffer += self.X[a]*r
		self.arm_selection_uploadbuffer[a] += 1

	def decide_arm(self):
		# self.theta_hat = np.linalg.inv(self.V_local).dot(self.b_local)
		self.theta_hat = np.linalg.solve(self.V_local, self.b_local)
		self.est_reward = self.X.dot(self.theta_hat)
		self.it = np.argmax(self.est_reward)
		self.jt = np.argmax(self.est_reward - self.est_reward[self.it] + 
							np.array([self.confidence_bound(self.X[self.it] - x, self.V_local) for x in self.X]))
		if self.jt == self.it:
			self.jt = np.argsort(self.est_reward - self.est_reward[self.it] + 
								np.array([self.confidence_bound(self.X[self.it] - x, self.V_local) for x in self.X]))[-2]
		y = self.X[self.it] - self.X[self.jt]
		arm_pull = np.argmin([y.dot(np.linalg.inv(self.V_local + self.matrix_dot(x))).dot(y) for x in (self.X)])
		r = (self.theta.dot(self.X[arm_pull]) + np.random.randn()*self.noise)
		self.localupdate(r, arm_pull)

	def uploadCommTrigger(self):
		trigger1 = np.linalg.det((self.V_local + self.V_uploadbuffer)) > (1+self.gamma1)*np.linalg.det(self.V_local)
		trigger2 = np.sum(self.arm_selection_local) + np.sum(self.arm_selection_uploadbuffer) > (1+self.gamma2)*np.sum(self.arm_selection_local)
		return (trigger1 or trigger2)

class DisALinPE:
	def __init__(self, dimension, epsilon, delta, NoiseScale, dataset, case, gap, n_clients, gamma1, gamma2, reg):
		self.dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.NoiseScale = NoiseScale
		self.dataset = dataset
		self.case = 'linear'
		self.n_clients = n_clients

		self.reg = reg
		self.gamma1 = gamma1
		self.gamma2 = gamma2

		# self.sigma = 1.0

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

		# records:
		self.totalCommCost = 0
		self.sampleComplexity = 0

		# client part
		self.clients = {}
		
		self.V_aggregated = np.eye(self.K)*self.reg
		self.b_aggregated = np.zeros(self.K)
		self.arm_selection_aggregated = np.zeros(self.K)

	def getReward(self, arm_vector):
		return self.theta.dot(arm_vector)
	
	def compute_true_gap(self):
		reward = np.zeros(self.K,dtype=float)
		for i in range(self.K):
			reward[i] = self.getReward(self.X[i])
		reward = np.sort(reward)
		gap = reward[self.K-1] - reward[self.K-2]
		print("Expected reward gap: ",gap)

	def matrix_dot(self, a):
		return np.expand_dims(a, axis=-1).dot(np.expand_dims(a, axis=0))

	def inilization(self):
		for i in range(self.K):
			self.V_aggregated += self.matrix_dot(self.X[i])
			r = (self.theta.dot(self.X[i]) + np.random.randn() * self.NoiseScale)
			self.b_aggregated += self.X[i]*r
			self.arm_selection_aggregated[i] += 1
			self.sampleComplexity += 1
		
		for currentclientID in range(self.n_clients):
			self.clients[currentclientID] = LocalClient(self.K, self.dimension, self.theta, self.NoiseScale, self.reg, self.X, self.gamma1, self.gamma2, self.n_clients)
			self.clients[currentclientID].V_local = copy.deepcopy(self.V_aggregated)
			self.clients[currentclientID].b_local = copy.deepcopy(self.b_aggregated)
			self.clients[currentclientID].arm_selection_local = copy.deepcopy(self.arm_selection_aggregated)

	def confidence_bound(self, x, V):
		delta = 0.05
		tmp = np.sqrt(x.dot(np.linalg.inv(V)).dot(x))
		C1 = (np.sqrt(2*self.gamma1)*self.n_clients)+ (np.sqrt(1+self.gamma1*self.n_clients))
		C2 = self.NoiseScale*(np.sqrt(self.dimension*np.log(2/delta*(1+(1+self.gamma2*self.n_clients)*np.sum(self.arm_selection_aggregated)/(min(self.gamma1, 1)*self.reg)))))
		return tmp*(np.sqrt(self.reg) + C1*C2)

	def run(self):
		
		self.compute_true_gap()

		self.inilization()



		for i in range(1000000):

			self.sampleComplexity += 1
			# if (self.sampleComplexity == 100):
			# 	exit(0)
			# print(self.sampleComplexity)
			# print(self.arm_selection_aggregated)
			# print()


			currentclientID = random.randint(0, self.n_clients-1)
			self.clients[currentclientID].decide_arm()

			if self.clients[currentclientID].uploadCommTrigger():
				self.totalCommCost += 2

				self.V_aggregated += self.clients[currentclientID].V_uploadbuffer
				self.b_aggregated += self.clients[currentclientID].b_uploadbuffer
				self.arm_selection_aggregated += self.clients[currentclientID].arm_selection_uploadbuffer


				# self.theta_hat = np.linalg.inv(self.V_aggregated).dot(self.b_aggregated)
				self.theta_hat = np.linalg.solve(self.V_aggregated, self.b_aggregated)
				self.est_reward = self.X.dot(self.theta_hat)
				self.it = np.argmax(self.est_reward)
				self.jt = np.argmax(self.est_reward - self.est_reward[self.it] + 
									np.array([self.confidence_bound(self.X[self.it] - x, self.V_aggregated) for x in self.X]))

				if self.jt == self.it:
					self.jt = np.argsort(self.est_reward - self.est_reward[self.it] + 
										np.array([self.confidence_bound(self.X[self.it] - x, self.V_aggregated) for x in self.X]))[-2]

				self.B = self.est_reward[self.jt] - self.est_reward[self.it] + self.confidence_bound(self.X[self.it] - self.X[self.jt], self.V_aggregated)

				if (self.B < self.epsilon):
					break
				
				self.clients[currentclientID].V_local = copy.deepcopy(self.V_aggregated)
				self.clients[currentclientID].b_local = copy.deepcopy(self.b_aggregated)
				self.clients[currentclientID].arm_selection_local = copy.deepcopy(self.arm_selection_aggregated)

				self.clients[currentclientID].V_uploadbuffer = np.zeros((self.dimension, self.K))
				self.clients[currentclientID].b_uploadbuffer = np.zeros(self.K)
				self.clients[currentclientID].arm_selection_uploadbuffer = np.zeros(self.K)
			
			if self.sampleComplexity % 5000 == 0:
				print("t: ", self.sampleComplexity)
				print("arm select: ", self.arm_selection_aggregated)
				print("totalComm: ", self.totalCommCost)
				print("B: ", self.B)
				print("E: ", self.epsilon)
				print("it jt: ", self.it, self.jt)
				print(self.est_reward[self.jt] - self.est_reward[self.it])
				print(self.confidence_bound(self.X[self.it] - self.X[self.jt], self.V_aggregated))
				print()
		best_arm = self.it
		print()
		print("SampleComplexity: ", self.sampleComplexity)
		print("arm selec: ", self.arm_selection_aggregated)
		print("totalComm: ", self.totalCommCost)
		print("B: ", self.B)
		print("E: ", self.epsilon)
		print("Best arm: ", best_arm)
		# exit(0)
		return self.sampleComplexity, self.arm_selection_aggregated, best_arm, self.totalCommCost