import copy
import numpy as np
from random import sample, shuffle
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_Linear_Case
from util_functions import featureUniform, gaussianFeature

from lib.LinGapE_sync import LinGapE_mult
from lib.LinGapE_full import LinGapE_full
from lib.LinGapE import LinGapE
from lib.DisALinPE import DisALinPE

class simulateOnlineData(object):
	def __init__(self, context_dimension, plot, dataset,
				 noise=lambda: 0, signature='', 
				 NoiseScale=0.0, poolArticleSize=None):

		self.simulation_signature = signature

		self.context_dimension = context_dimension
		self.batchSize = 1

		self.plot = plot

		self.noise = noise

		self.NoiseScale = NoiseScale
		self.dataset = str(dataset)
		

		if poolArticleSize is None:
			self.poolArticleSize = len(self.articles)
		else:
			self.poolArticleSize = poolArticleSize


	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M')
		filenameWriteSampleComplex = os.path.join(save_Linear_Case, 'SampleComplex_dataset' + self.dataset + timeRun + '.csv') 
		filenameWriteCommCost = os.path.join(save_Linear_Case, 'AccCommCost_dataset' + self.dataset + timeRun + '.csv')


		# SampleComList_U = {}
		SampleComList_U_Hom = {}
		# SampleComList_L = {}
		SampleComList_L_Hom = {}
		SampleComList_D_Hom = {}

		CommCostList_L = {}
		CommCostList_U = {}
		CommCostList_D = {}

		for alg_name, alg in algorithms.items():
			# if alg_name[17] == 'U' and alg_name[13:16] == 'Sin':
			# 	SampleComList_U[alg_name] = []
			# if alg_name[17] == 'L' and alg_name[13:16] == 'Sin':
			# 	SampleComList_L[alg_name] = []
			if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
				SampleComList_L_Hom[alg_name] = []
				CommCostList_L[alg_name] = []
			if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
				SampleComList_U_Hom[alg_name] = []
				CommCostList_U[alg_name] = []
			if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
				SampleComList_D_Hom[alg_name] = []
				CommCostList_D[alg_name] = []
			# print(alg_name[17])
			# print(alg_name[13:16])
		# print(SampleComList_U)
		# exit(0)

		for alg_name, alg in algorithms.items():
			print('starting algorithm ' + alg_name + ' on dataset ' + str(alg.dataset))
			
			samplecomplexity, arm_selection, best_arm, totalCommCost = alg.run()

			# if alg_name[17] == 'U' and alg_name[13:16] == 'Sin':
			# 	SampleComList_U[alg_name].append(samplecomplexity)
			# if alg_name[17] == 'L' and alg_name[13:16] == 'Sin':
			# 	SampleComList_L[alg_name].append(samplecomplexity)
			if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
				SampleComList_L_Hom[alg_name].append(samplecomplexity)
				CommCostList_L[alg_name].append(totalCommCost)
			if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
				SampleComList_U_Hom[alg_name].append(samplecomplexity)
				CommCostList_U[alg_name].append(totalCommCost)
			if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
				SampleComList_D_Hom[alg_name].append(samplecomplexity)
				CommCostList_D[alg_name].append(totalCommCost)


				

		with open(filenameWriteSampleComplex, 'w') as f:
			f.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')
		for alg_name in algorithms.keys():
			if alg_name[13:16] == 'Hom':
				with open(filenameWriteCommCost, 'w') as f:
					f.write(','.join([str(alg_name)]))
					f.write('\n')

		print()
		print('Finish running, runtime: ', datetime.datetime.now() - self.startTime)
		# print('arm selection: ', arm_selection)
		# print('SampleComplexity: ',samplecomplexity)
		# print('Best Arm: ', best_arm)
		
		# Initialization
		# userSize = len(self.users)


		with open(filenameWriteSampleComplex, 'a+') as f:
			for alg_name in algorithms.keys():
				# if alg_name[17] == 'U' and alg_name[13:16] == 'Sin':
				# 	f.write(','. join([str(SampleComList_U[alg_name][-1])]))
				# 	f.write('\n')
				# if alg_name[17] == 'L' and alg_name[13:16] == 'Sin':
				# 	f.write(','. join([str(SampleComList_L[alg_name][-1])]))
					# f.write('\n')
				if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(SampleComList_L_Hom[alg_name][-1])]))
					f.write('\n')
				if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(SampleComList_U_Hom[alg_name][-1])]))
					f.write('\n')
				if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(SampleComList_D_Hom[alg_name][-1])]))
					f.write('\n')
		with open(filenameWriteCommCost, 'a+') as f:
			for alg_name in algorithms.keys():
				if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(CommCostList_L[alg_name][-1])]))
					f.write('\n')
				if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(CommCostList_U[alg_name][-1])]))
					f.write('\n')
				if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
					f.write(','. join([str(CommCostList_D[alg_name][-1])]))
					f.write('\n')
	
		if (self.plot==True): # only plot
			# # plot the results
			fig, axa = plt.subplots(2, 1, sharex='all')
			# Remove horizontal space between axes
			fig.subplots_adjust(hspace=0)

			sx = []
			# syl = []
			# syu = []
			sylh = []
			syuh = []
			sydh = []

			print("=====Sample Complexity=====")
			for alg_name in algorithms.keys():
				# if alg_name[17] == 'U' and alg_name[13:16] == 'Sin':
				# 	syu.append(SampleComList_U[alg_name])
				# 	sx.append(float(alg_name[5])/10)
				# 	print('%s: %.2f' % (alg_name,SampleComList_U[alg_name][-1]))
				# if alg_name[17] == 'L' and alg_name[13:16] == 'Sin':
				# 	syl.append(SampleComList_L[alg_name])
				# 	print('%s: %.2f' % (alg_name, SampleComList_L[alg_name][-1]))
				if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
					sx.append(float(alg_name[5])/10)
					sylh.append(SampleComList_L_Hom[alg_name])
					print('%s: %.2f' % (alg_name, SampleComList_L_Hom[alg_name][-1]))
				if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
					syuh.append(SampleComList_U_Hom[alg_name])
					print('%s: %.2f' % (alg_name, SampleComList_U_Hom[alg_name][-1]))
				if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
					sydh.append(SampleComList_D_Hom[alg_name])
					print('%s: %.2f' % (alg_name, SampleComList_D_Hom[alg_name][-1]))

			# axa[0].plot(sx, syu, marker='o', linestyle='dotted', label='UGapE')
			# axa[0].plot(sx, syl, marker='o', linestyle='dotted', label='LinGapE')
			axa[0].plot(sx, sylh, marker='v', linestyle='dotted', label='LinGapE-Sync')
			axa[0].plot(sx, syuh, marker='o', linestyle='dotted', label='LinGapE-Single')
			axa[0].plot(sx, sydh, marker='s', linestyle='dotted', label='FedALinPE')
			axa[0].legend(loc='upper right')
			axa[0].set_xlabel("Expected Reward Gap")
			axa[0].set_ylabel("SampleComplexity")

			cx = []
			cyl = []
			cyu = []
			cyd = []
			print("=====Comm Cost=====")
			for alg_name in algorithms.keys():
				if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
					cx.append(float(alg_name[5])/10)
					cyl.append(CommCostList_L[alg_name])
					print('%s: %.2f' % (alg_name, CommCostList_L[alg_name][-1]))
				if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
					cyu.append(CommCostList_U[alg_name])
					print('%s: %.2f' % (alg_name, CommCostList_U[alg_name][-1]))
				if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
					cyd.append(CommCostList_D[alg_name])
					print('%s: %.2f' % (alg_name, CommCostList_D[alg_name][-1]))
			axa[1].plot(cx, cyl, marker='v', linestyle='dotted', label='LinGapE-Sync')
			# axa[1].plot(cx, cyu, marker='o', linestyle='dotted', label='LinGapE-Single')
			axa[1].plot(cx, cyd, marker='s', linestyle='dotted', label='FedALinPE')
			axa[1].set_xlabel("Expected Reward Gap")
			axa[1].set_ylabel("Communication Cost")
			axa[1].legend(loc='upper right')
			plt.savefig(os.path.join(save_Linear_Case, "SamConAndCommCost_dataset" + self.dataset + str(timeRun) + '.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.0)
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
		n_users = 10
	
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
		dataset = 0
	
	# set default case to linear case
	if args.DC:
		case = args.DC
	else:
		case = 'linear'
	
	NoiseScale = 0.3  # standard deviation of Gaussian noise
	n_articles = 5
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
										poolArticleSize=poolArticleSize,
										dataset=dataset)
	
	# based on the following caculation, gamma1 = 0.01, gamma2 = 0.00632455532033676, reg = 22.378511198424423
	# gamma1 = 1/n_users/n_users
	# gamma2 = 1/(np.sqrt(n_users*n_users*n_users)*5)
	# tmp = (np.sqrt(1+gamma1*n_users)+np.sqrt(2*gamma1)*n_users)
	# reg = NoiseScale*NoiseScale*tmp*tmp*np.log(2/0.05)

	# gamma1 = 0.01
	# gamma2 = 0.005
	# tmp = (np.sqrt(1+gamma1*n_users)+np.sqrt(2*gamma1)*n_users)
	# reg = NoiseScale*NoiseScale*tmp*tmp*np.log(2/0.05)
	# reg = 10wdad

	# noise = 1
	# gamma1 = 0.005
	# gamma2 = 0.001
	# # tmp = (np.sqrt(1+gamma1*n_users)+np.sqrt(2*gamma1)*n_users)
	# # reg = NoiseScale*NoiseScale*tmp*tmp*np.log(2/0.05)
	# reg = 5

	# noise = 0.3
	gamma1 = 0.04
	# gamma2 = 1/(np.sqrt(n_users*n_users*n_users)*5)
	gamma2 = 0.1
	tmp = (np.sqrt(1+gamma1*n_users)+np.sqrt(2*gamma1)*n_users)
	reg = NoiseScale*NoiseScale*tmp*tmp*np.log(2/0.05)

	# print(gamma1)
	# print(gamma2)
	print(reg)
	# exit(0)

	## Initiate Bandit Algorithms ##
	algorithms = {}
	if dataset == 0:
		# algorithms['gap=.1_data0_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=1)
		# algorithms['gap=.2_data0_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=2)
		# algorithms['gap=.3_data0_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=3)
		# algorithms['gap=.4_data0_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=4)
		# algorithms['gap=.5_data0_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=5)
		

		# algorithms['gap=.1_data0_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=1)
		# algorithms['gap=.2_data0_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=2)
		# algorithms['gap=.3_data0_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=3)
		# algorithms['gap=.4_data0_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=4)
		# algorithms['gap=.5_data0_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=0, case='linear', gap=5)
		
		algorithms['gap=.1_data0_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=1)
		algorithms['gap=.2_data0_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=2)
		algorithms['gap=.3_data0_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=3)
		algorithms['gap=.4_data0_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=4)
		algorithms['gap=.5_data0_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=5)
		
		algorithms['gap=.1_data0_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=1)
		algorithms['gap=.2_data0_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=2)
		algorithms['gap=.3_data0_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=3)
		algorithms['gap=.4_data0_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=4)
		algorithms['gap=.5_data0_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=0, case='linear', gap=5)

	if dataset == 2:
		# algorithms['gap=.1_data2_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=1)
		# algorithms['gap=.2_data2_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=2)
		# algorithms['gap=.3_data2_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=3)
		# algorithms['gap=.4_data2_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=4)
		# algorithms['gap=.5_data2_Sin_U'] = UGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=5)
		

		# algorithms['gap=.1_data2_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=1)
		# algorithms['gap=.2_data2_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=2)
		# algorithms['gap=.3_data2_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=3)
		# algorithms['gap=.4_data2_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=4)
		# algorithms['gap=.5_data2_Sin_L'] = LinGapE(dimension=5, epsilon=epsilon, 
		# 						delta= delta, NoiseScale=NoiseScale, 
		# 						dataset=2, case='linear', gap=5)
		
		algorithms['gap=.1_data2_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=1)
		algorithms['gap=.2_data2_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=2)
		algorithms['gap=.3_data2_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=3)
		algorithms['gap=.4_data2_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=4)
		algorithms['gap=.5_data2_Hom_L'] = LinGapE_mult(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=5)
		
		algorithms['gap=.1_data2_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=1)
		algorithms['gap=.2_data2_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=2)
		algorithms['gap=.3_data2_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=3)
		algorithms['gap=.4_data2_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=4)
		algorithms['gap=.5_data2_Hom_U'] = LinGapE_full(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=2, case='linear', gap=5)
		
		algorithms['gap=.1_data2_Hom_D'] = DisALinPE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=dataset, case=case, gap=1, n_clients=n_users, 
								gamma1=gamma1, gamma2=gamma2, reg=reg)
		algorithms['gap=.2_data2_Hom_D'] = DisALinPE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=dataset, case=case, gap=2, n_clients=n_users, 
								gamma1=gamma1, gamma2=gamma2, reg=reg)
		algorithms['gap=.3_data2_Hom_D'] = DisALinPE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=dataset, case=case, gap=3, n_clients=n_users, 
								gamma1=gamma1, gamma2=gamma2, reg=reg)
		algorithms['gap=.4_data2_Hom_D'] = DisALinPE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=dataset, case=case, gap=4, n_clients=n_users, 
								gamma1=gamma1, gamma2=gamma2, reg=reg)
		algorithms['gap=.5_data2_Hom_D'] = DisALinPE(dimension=5, epsilon=epsilon, 
								delta= delta, NoiseScale=NoiseScale, 
								dataset=dataset, case=case, gap=5, n_clients=n_users, 
								gamma1=gamma1, gamma2=gamma2, reg=reg)



	## Run Simulation ##
	print("Starting")
	simExperiment.runAlgorithms(algorithms)