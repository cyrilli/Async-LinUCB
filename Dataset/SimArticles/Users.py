import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None):
		self.id = id
		self.theta = theta

class UserManager():
	def __init__(self, dimension, userNum, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def simulateThetaForHomoUsers(self):
		users = []
		thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
		l2_norm = np.linalg.norm(thetaVector, ord=2)
		thetaVector = thetaVector/l2_norm
		for key in range(self.userNum):
			users.append(User(key, thetaVector))

		return users

	def saveHomoUsers(self, users, filename):
		with open(filename, 'w') as f:
			for i in range(len(users)):
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
	
	def loadHomoUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				uid, theta = json.loads(line)
				users.append(User(uid, np.array(theta)))
		return users

	def simulateThetaForHeteroUsers(self, global_dim):
		local_dim = self.dimension-global_dim
		users = []
		thetaVector_g = self.thetaFunc(global_dim, argv=self.argv)
		l2_norm = np.linalg.norm(thetaVector_g, ord=2)
		thetaVector_g = thetaVector_g/l2_norm
		for key in range(self.userNum):
			thetaVector_l = self.thetaFunc(local_dim, argv=self.argv)
			l2_norm = np.linalg.norm(thetaVector_l, ord=2)
			thetaVector_l = thetaVector_l/l2_norm

			thetaVector = np.concatenate([thetaVector_g, thetaVector_l])
			users.append(User(key, thetaVector))

		return users

