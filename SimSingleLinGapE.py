import copy
import numpy as np
from random import sample, shuffle
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager

from lib.AsyncLinUCB import AsyncLinUCB
from lib.SyncLinUCB import SyncLinUCB
from lib.LinGapE import LinGapE

class simulateOnlineData(object):
	def __init__(self, context_dimension, plot, articles,
				 users, noise=lambda: 0, signature='', NoiseScale=0.0, poolArticleSize=None):

		self.simulation_signature = signature

		self.context_dimension = context_dimension
		self.batchSize = 1

		self.plot = plot

		self.noise = noise

		self.NoiseScale = NoiseScale
		
		self.articles = articles
		self.users = users

		if poolArticleSize is None:
			self.poolArticleSize = len(self.articles)
		else:
			self.poolArticleSize = poolArticleSize


	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M')
		filenameWriteSampleComplex = os.path.join(save_address, 'SampleComplex' + timeRun + '.csv') 

		samplecomplexity = 0

		for alg_name, alg in algorithms.items():
			print('starting algorithm ' + alg_name + ' on dataset ' + str(alg.dataset))
			samplecomplexity, arm_selection = alg.run(self.articles)

		print()
		print('Finish running, runtime: ', datetime.datetime.now() - self.startTime)
		print('arm selection: ', arm_selection)
		print('SampleComplexity: ',samplecomplexity)
		
		# Initialization
		# userSize = len(self.users)


		# with open(filenameWriteSampleComplex, 'w') as f:
		# 	f.write(','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.keys()]))
		
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--E', dest='E', help='Target accuracy')
	parser.add_argument('--D', dest='D', help='Target confidence level')
	parser.add_argument('--N', dest='N', help='total number of clients')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--Dataset', dest='Data', help='Choose dataset, 0 for articles, 1 for data setting1, 2 for data setting 2')
	parser.add_argument('--Data_Case', dest='DC', type=str, help='Choose dataset case, linear for linear case, tabular for tabular case')
	args = parser.parse_args()

	## Environment Settings ##
	if args.contextdim:
		context_dimension = int(args.contextdim)
	else:
		context_dimension = 5

	if args.N:
		n_users = int(args.N)
	else:
		n_users = 100
	
	if args.E:
		epsilon = float(args.E)
	else:
		epsilon =2*(1 - np.cos(0.01))
	
	if args.D:
		delta = float(args.D)
	else:
		delta = 0.05
	
	if args.Data:
		dataset = int(args.Data)
	else:
		dataset = 2
	
	# set default case to linear case
	if args.DC:
		case = args.DC
	else:
		case = 'linear'
	
	NoiseScale = 0.1  # standard deviation of Gaussian noise
	n_articles = 10
	poolArticleSize = 25

	## Set Up Simulation ##
	UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
	users = UM.simulateThetaForHomoUsers()
	AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
	articles = AM.simulateArticlePool()

	simExperiment = simulateOnlineData(	context_dimension=context_dimension,
										plot=True,
										articles=articles,
										users = users,
										noise=lambda: np.random.normal(scale=NoiseScale),
										signature=AM.signature,
										NoiseScale=NoiseScale,
										poolArticleSize=poolArticleSize)

	## Initiate Bandit Algorithms ##
	algorithms = {}

	algorithms['testLinGapE'] = LinGapE(dimension=context_dimension, epsilon=epsilon, 
				     					delta= delta, NoiseScale=NoiseScale, articles=articles,
										dataset=dataset, case=case, gap=1)

	## Run Simulation ##
	print("Starting")
	simExperiment.runAlgorithms(algorithms)