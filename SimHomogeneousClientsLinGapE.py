import copy
import numpy as np
from random import sample, shuffle
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address, save_LinGapE_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager

from lib.LinGapE_multi import LinGapE

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
		filenameWriteSampleComplex = os.path.join(save_LinGapE_address, 'SampleComplex' + timeRun + '.csv') 
		filenameWriteCommCost = os.path.join(save_LinGapE_address, 'AccCommCost' + timeRun + '.csv')

		SampleComList = {}
		CommCostList = {}

		for alg_name, alg in algorithms.items():
			SampleComList[alg_name] = []
			CommCostList[alg_name] = []

		with open(filenameWriteCommCost, 'w') as f:
			f.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')
		
		with open(filenameWriteSampleComplex, 'w') as f:
			f.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		for alg_name, alg in algorithms.items():
			print('starting algorithm ' + alg_name + ' on dataset ' + str(alg.dataset))
			samplecomplexity, totalCommCost = alg.run(self.articles)
			print('samplecomplexity: ',samplecomplexity)
			print('totalCommCost: ',totalCommCost)
			print()
			SampleComList[alg_name].append(samplecomplexity)
			CommCostList[alg_name].append(totalCommCost)


		with open(filenameWriteCommCost, 'a+') as f:
			f.write(','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
			f.write('\n')
		
		with open(filenameWriteSampleComplex, 'a+') as f:
			f.write(','. join([str(SampleComList[alg_name][-1]) for alg_name in algorithms.keys()]))
			f.write('\n')

		if (self.plot==True): # only plot
			# # plot the results
			fig, axa = plt.subplots(2, 1, sharex='all')
			# Remove horizontal space between axes
			fig.subplots_adjust(hspace=0)

			sx = []
			sy = []
			print("=====Sample Complexity=====")
			for alg_name in algorithms.keys():
				sx.append(int(alg_name[4]))
				sy.append(SampleComList[alg_name])
				print('%s: %.2f' % (alg_name, SampleComList[alg_name][-1]))

			axa[0].plot(sx, sy, marker='o', linestyle='dotted')
			axa[0].set_xlabel("dimesions")
			axa[0].set_ylabel("SampleComplexity")

			cx = []
			cy = []
			print("=====Comm Cost=====")
			for alg_name in algorithms.keys():
				cx.append(int(alg_name[4]))
				cy.append(CommCostList[alg_name])
				print('%s: %.2f' % (alg_name, CommCostList[alg_name][-1]))
			axa[1].plot(cx, cy, marker='o', linestyle='dotted')
			axa[1].set_xlabel("dimesions")
			axa[1].set_ylabel("Communication Cost")
			plt.savefig(os.path.join(save_LinGapE_address, "SamComAndcommCost" + "_" + str(timeRun) + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.0)
			plt.show()
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--E', dest='E', help='Target accuracy')
	parser.add_argument('--D', dest='D', help='Target confidence level')
	parser.add_argument('--N', dest='N', help='total number of clients')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--Dataset', dest='Data', help='Choose dataset, 0 for articles, 1 for data setting1, 2 for data setting 2')
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
		dataset = 1
	
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

	# algorithms['testLinGapE'] = LinGapE(dimension=context_dimension, epsilon=epsilon, 
	# 			     					delta= delta, NoiseScale=NoiseScale, articles=articles,
	# 									dataset=dataset)
	
	algorithms['dim=2_data2'] = LinGapE(dimension=2, epsilon=epsilon, 
									delta= delta, NoiseScale=NoiseScale, articles=articles,
									dataset=2)
	algorithms['dim=5_data2'] = LinGapE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, articles=articles,
								dataset=2)
	algorithms['dim=8_data2'] = LinGapE(dimension=8, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, articles=articles,
								dataset=2)

	## Run Simulation ##
	print("Starting")
	simExperiment.runAlgorithms(algorithms)