import numpy as np
import copy
import random


class LocalClient:
	def __init__(self, K, featureDimension, t_reward, sigma, noise, n_clients):
		self.K = K
		self.d = featureDimension
		self.t_reward = t_reward
		self.sigma = sigma
		self.noise = noise
		self.n_clients = n_clients

		self.est_reward_local = np.zeros(self.K)
		self.arm_selection_local = np.zeros(self.K)

		self.est_reward_uploadbuffer = np.zeros(self.K)
		self.arm_selection_uploadbuffer = np.zeros(self.K)

	def confidence_bound(self, x, gamma):
		delta = 0.05
		tmp = (1 + gamma*self.n_clients)*np.sum(self.arm_selection_local)
		return self.noise * np.sqrt(2/self.arm_selection_local[x] * np.log(4*self.K/delta * tmp * tmp))

	def get_reward(self, mu):
		# return np.random.normal(mu, self.sigma)
		return mu

	# update local paramters based on the selected arm
	def localUpdate(self, a):
		self.est_reward_local[a] += self.get_reward(self.t_reward[a])
		self.arm_selection_local[a] += 1

		self.est_reward_uploadbuffer[a] += self.get_reward(self.t_reward[a])
		self.arm_selection_uploadbuffer[a] += 1

	def decide_arm(self, gamma):
		self.it = np.argmax(self.est_reward_local)
		self.jt = np.argmax(self.est_reward_local - 
		      	  self.est_reward_local[self.it] + 
				  (np.array([self.confidence_bound(x, gamma) + self.confidence_bound(self.it, gamma) for x in range(self.K)])))
		if self.it == self.jt:
			self.jt = np.argsort(self.est_reward_local - 
								 self.est_reward_local[self.it] + 
								 (np.array([self.confidence_bound(x, gamma) + self.confidence_bound(self.it, gamma) for x in range(self.K)])))[-2]
		if self.confidence_bound(self.it, gamma) > self.confidence_bound(self.jt, gamma):
			self.localUpdate(self.it)
		else:
			self.localUpdate(self.jt)
	
	# receive the estimate reward and arm selections fromn server
	def localReceive(self, est_reward, arm_selection):
		self.est_reward_local = copy.deepcopy(est_reward)
		self.arm_selection_local = copy.deepcopy(arm_selection)

		self.est_reward_uploadbuffer = copy.deepcopy(est_reward)
		self.arm_selection_uploadbuffer = copy.deepcopy(arm_selection)

	def uploadCommTrigger(self, server_sample_complexity, gamma):
		local_sampleComplexity = np.sum(self.arm_selection_local)
		return server_sample_complexity + local_sampleComplexity > (1+gamma)*server_sample_complexity

	


class DisTarPE:
	def __init__(self, dimension, epsilon, delta, NoiseScale, dataset, case, gap, n_clients, gamma):
		self. dimension = dimension
		self.epsilon = epsilon
		self.delta = delta
		self.Noise = NoiseScale
		self.dataset = dataset
		self.case = 'Tabular'
		self.n_clients = n_clients

		self.sigma = 1.0

		# for computation
		if dataset == 2 and case == 'tabular':
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
		self.totalCommCost = 0
		self.sampleComplexity = 0

		# client part
		self.clients = {}
		# aggregated sufficient statistics of all clients
		self.est_reward_aggregated = np.zeros(self.K)
		# self.b_aggregated = np.zeros(self.dimension)
		self.arm_selection_aggregated = np.zeros(self.K) # for inilization, all arm got pulled once
		

		# aggregated sufficient statistics that haven't been sent to each client
		self.est_reward_downloadbuffer = {}
		# self.b_downloadbuffer = {}
		self.arm_selection_downloadbuffer = {}

		# trigger parameters
		self.gamma = gamma

	
	def compute_true_gap(self):
		gap = self.t_reward[0] - self.t_reward[1]
		print("Expected reward gap: ",gap)

	def get_reward(self, mu):
		# return np.random.normal(mu, self.sigma)
		return mu

	def inilization(self):
		for i in range(self.K):
			r = self.get_reward(self.t_reward[i])
			# r = self.t_reward[i] + np.random.randn() *self.sigma
			self.est_reward_aggregated[i] = r
			self.arm_selection_aggregated[i] += 1
			self.sampleComplexity += 1
		
		for currentclientID in range(self.n_clients):
			self.clients[currentclientID] = LocalClient(self.K, self.dimension, self.t_reward, self.sigma, self.Noise, self.n_clients)
			self.est_reward_downloadbuffer[currentclientID] = copy.deepcopy(self.est_reward_aggregated)
			self.arm_selection_downloadbuffer[currentclientID] = copy.deepcopy(self.arm_selection_aggregated)

			self.clients[currentclientID].localReceive(self.est_reward_aggregated, self.arm_selection_aggregated)

	def confidence_bound(self, x, gamma):
		delta = 0.05
		tmp = (1 + gamma*self.n_clients)*np.sum(self.arm_selection_aggregated)
		return self.Noise * np.sqrt(2/self.arm_selection_aggregated[x] * np.log(4*self.K/delta*tmp*tmp))

	def est_reward_update(self, clientID):
		tmp_reward = self.est_reward_aggregated * self.arm_selection_aggregated + self.clients[clientID].est_reward_uploadbuffer
		tmp_arm_pull = self.arm_selection_aggregated + self.clients[clientID].arm_selection_uploadbuffer
		return tmp_reward/tmp_arm_pull

	def run(self):

		# compute the true gap(expected reward gap)
		self.compute_true_gap()

		# initialize by pulling each arm once at first
		self.inilization()

		# Pull for the K+1 iteration
		# KP1clientID = random.randint(0, self.n_clients)
		# self.clients[KP1clientID].decide_arm(self.gamma)
		# Communication for caculate B 
		# self.totalCommCost += 1

		# Caculate B based on the initialization
		self.it = np.argmax(self.est_reward_aggregated)
		self.jt = np.argmax(self.est_reward_aggregated - 
		      	  self.est_reward_aggregated[self.it] + 
				  np.array([self.confidence_bound(x, self.gamma) + self.confidence_bound(self.it, self.gamma) for x in range(self.K)]))
		if self.it == self.jt:
			self.jt = np.argsort(self.est_reward_aggregated - 
								 self.est_reward_aggregated[self.it] + 
								 np.array([self.confidence_bound(x, self.gamma) + self.confidence_bound(self.it, self.gamma) for x in range(self.K)]))[-2]
		self.B = self.est_reward_aggregated[self.jt] - self.est_reward_aggregated[self.it] + self.confidence_bound(self.it, self.gamma) + self.confidence_bound(self.jt, self.gamma)
		# print(self.est_reward_aggregated[self.jt])
		# print(self.est_reward_aggregated[self.it])
		# print(self.confidence_bound(self.it, self.gamma))
		# print(self.confidence_bound(self.jt, self.gamma))
		
		# exit(0)
		t = self.K
		# print(self.est_reward_aggregated)
		# exit(0)
		while(self.B > self.epsilon):
			currentclientID = random.randint(0, self.n_clients-1)
			self.clients[currentclientID].decide_arm(self.gamma)

			# for i in range(self.n_clients):
			# 	print(self.clients[i].arm_selection_local)
			# exit(0)

			self.sampleComplexity += 1

			t += 1
			if t%5000 == 0:
				print("t:", t)
				print(self.est_reward_aggregated)
				print("arm_select:",self.arm_selection_aggregated)
				print("B:", self.B)
				print("E:", self.epsilon)
				print("it, jt: ", self.it, self.jt)
				print()
			# print(self.clients[currentclientID].uploadCommTrigger(self.sampleComplexity, self.gamma))
			if self.clients[currentclientID].uploadCommTrigger(np.sum(self.arm_selection_aggregated), self.gamma):
				self.totalCommCost += 1

				self.est_reward_aggregated = self.est_reward_update(currentclientID)
				# print(self.arm_selection_aggregated)
				self.arm_selection_aggregated += self.clients[currentclientID].arm_selection_uploadbuffer
				# self.sampleComplexity = np.sum(self.arm_selection_aggregated)

				self.it = np.argmax(self.est_reward_aggregated)
				self.jt = np.argmax(self.est_reward_aggregated - 
									self.est_reward_aggregated[self.it] + 
									np.array([self.confidence_bound(x, self.gamma) + self.confidence_bound(self.it, self.gamma) for x in range(self.K)]))
				if (self.jt == self.it):
					self.jt = np.argsort(self.est_reward_aggregated - 
										 self.est_reward_aggregated[self.it] + 
										 np.array([self.confidence_bound(x, self.gamma) + self.confidence_bound(self.it, self.gamma) for x in range(self.K)]))[-2]
				
				self.B = self.est_reward_aggregated[self.jt] - self.est_reward_aggregated[self.it] + self.confidence_bound(self.it, self.gamma) + self.confidence_bound(self.jt, self.gamma)
				# print("B:", self.B)
				# print()
				# update server's download buffer for other clients
				for clientID in self.arm_selection_downloadbuffer.keys():
					if clientID != currentclientID:
						self.est_reward_downloadbuffer[clientID] = self.est_reward_update(clientID)
						self.arm_selection_downloadbuffer[clientID] += self.clients[clientID].arm_selection_uploadbuffer

				# clear client's upload buffer
				self.clients[currentclientID].est_reward_uploadbuffer = np.zeros(self.K)
				self.clients[currentclientID].arm_selection_uploadbuffer = np.zeros(self.K)
				
				for clientID, client in self.clients.items():
					self.totalCommCost += 1
					client.est_reward_local += self.est_reward_downloadbuffer[clientID]
					client.arm_selection_local += self.arm_selection_downloadbuffer[clientID]

					self.est_reward_downloadbuffer[clientID] = np.zeros(self.K)
					self.arm_selection_downloadbuffer[clientID] = np.zeros(self.K)
		
		best_arm = self.it
		print()
		print("T:", self.sampleComplexity)
		print("arm_select:",self.arm_selection_aggregated)
		# print("B:", self.B)
		# print("E:", self.epsilon)
		print("Best arm: ", best_arm)
		print("totalCommcost:", self.totalCommCost)
		exit(0)
		return self.sampleComplexity, self.arm_selection_aggregated, best_arm, self.totalCommCost