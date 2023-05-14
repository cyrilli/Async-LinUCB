import copy
import numpy as np
from random import sample, shuffle
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_Tabular_Case
from util_functions import featureUniform, gaussianFeature

from lib.AsyncLinUCB import AsyncLinUCB
from lib.SyncLinUCB import SyncLinUCB
from lib.UGapE import UGapE
from lib.LinGapE import LinGapE

class simulateOnlineData(object):
	def __init__(self, context_dimension, plot, 
				 noise=lambda: 0, signature='', 
				 NoiseScale=0.0, poolArticleSize=None):

		self.simulation_signature = signature

		self.context_dimension = context_dimension
		self.batchSize = 1

		self.plot = plot

		self.noise = noise

		self.NoiseScale = NoiseScale
		

		if poolArticleSize is None:
			self.poolArticleSize = len(self.articles)
		else:
			self.poolArticleSize = poolArticleSize


	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M')
		filenameWriteSampleComplex = os.path.join(save_Tabular_Case, 'SampleComplex' + timeRun + '.csv') 

		SampleComList_U = {}
		SampleComList_L = {}

		for alg_name, alg in algorithms.items():
			if alg_name[17] == 'U':
				SampleComList_U[alg_name] = []
			if alg_name[17] == 'L':
				SampleComList_L[alg_name] = []


		for alg_name, alg in algorithms.items():
			print('starting algorithm ' + alg_name + ' on dataset ' + str(alg.dataset))
			samplecomplexity, arm_selection, best_arm = alg.run()
			if alg_name[17] == 'U':
				SampleComList_U[alg_name].append(samplecomplexity)
			if alg_name[17] == 'L':
				SampleComList_L[alg_name].append(samplecomplexity)

		with open(filenameWriteSampleComplex, 'w') as f:
			f.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		print()
		print('Finish running, runtime: ', datetime.datetime.now() - self.startTime)
		print('arm selection: ', arm_selection)
		print('SampleComplexity: ',samplecomplexity)
		print('Best Arm: ', best_arm)
		
		# Initialization
		# userSize = len(self.users)


		with open(filenameWriteSampleComplex, 'a+') as f:
			for alg_name in algorithms.keys():
				if alg_name[17] == 'U':
					f.write(','. join([str(SampleComList_U[alg_name][-1])]))
					f.write('\n')
				if alg_name[17] == 'L':
					f.write(','. join([str(SampleComList_L[alg_name][-1])]))
					f.write('\n')
	
		if (self.plot==True): # only plot
			# # plot the results
			fig, axa = plt.subplots(1, 1, sharex='all')
			# Remove horizontal space between axes
			# fig.subplots_adjust(hspace=0)
			case = alg_name[13:15]

			sx = []
			syl = []
			syu = []

			print("=====Sample Complexity=====")
			for alg_name in algorithms.keys():
				if alg_name[17] == 'U':
					syu.append(SampleComList_U[alg_name])
					sx.append(float(alg_name[5])/10)
					print('%s: %.2f' % (alg_name,SampleComList_U[alg_name][-1]))
				if alg_name[17] == 'L':
					syl.append(SampleComList_L[alg_name])
					print('%s: %.2f' % (alg_name, SampleComList_L[alg_name][-1]))

			axa.plot(sx, syu, marker='o', linestyle='dotted', label='UGapE')
			axa.plot(sx, syl, marker='o', linestyle='dotted', label='LinGapE')
			axa.set_xlabel("Expected Reward Gap")
			axa.set_ylabel("SampleComplexity")
			axa.legend()
			plt.savefig(os.path.join(save_Tabular_Case, "SamCon" + "_" + str(timeRun) + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.0)
			plt.show()

		

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
	# UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
	# users = UM.simulateThetaForHomoUsers()
	# AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
	# articles = AM.simulateArticlePool()

	simExperiment = simulateOnlineData(	context_dimension=context_dimension,
										plot=True,
										noise=lambda: np.random.normal(scale=NoiseScale),
										NoiseScale=NoiseScale,
										poolArticleSize=poolArticleSize)

	## Initiate Bandit Algorithms ##
	algorithms = {}

	algorithms['gap=.1_data2_tar_U'] = UGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=1)
	algorithms['gap=.2_data2_tar_U'] = UGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=2)
	algorithms['gap=.3_data2_tar_U'] = UGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=3)
	algorithms['gap=.4_data2_tar_U'] = UGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=4)
	algorithms['gap=.5_data2_tar_U'] = UGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=5)
	

	algorithms['gap=.1_data2_tar_L'] = LinGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=1)
	algorithms['gap=.2_data2_tar_L'] = LinGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=2)
	algorithms['gap=.3_data2_tar_L'] = LinGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=3)
	algorithms['gap=.4_data2_tar_L'] = LinGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=4)
	algorithms['gap=.5_data2_tar_L'] = LinGapE(dimension=5, epsilon=epsilon, 
							delta= delta, NoiseScale=NoiseScale, 
							dataset=2, case='tabular', gap=5)


	## Run Simulation ##
	print("Starting")
	simExperiment.runAlgorithms(algorithms)