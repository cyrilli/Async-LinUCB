from matplotlib import pyplot as plt
import os
import sys
import copy
sys.path.append('/nfs/stak/users/songchen/research/Async-LinUCB')
from conf import sim_files_folder, save_Linear_Case, save_Linear_Movie_Case

from lib.LinGapE_sync import LinGapE_mult
from lib.LinGapE_full import LinGapE_full
from lib.LinGapE import LinGapE
from lib.DisALinPE import DisALinPE


algorithms = {}
SampleComList_L_Hom = []
SampleComList_U_Hom = []
SampleComList_D_Hom = []
CommCostList_L = []
CommCostList_U = []
CommCostList_D = []

algorithms['gap=.1_data2_Hom_U'] = 0
algorithms['gap=.2_data2_Hom_U'] = 0
algorithms['gap=.3_data2_Hom_U'] = 0
algorithms['gap=.4_data2_Hom_U'] = 0
algorithms['gap=.5_data2_Hom_U'] = 0

algorithms['gap=.1_data2_Hom_L'] = 0
algorithms['gap=.2_data2_Hom_L'] = 0
algorithms['gap=.3_data2_Hom_L'] = 0
algorithms['gap=.4_data2_Hom_L'] = 0
algorithms['gap=.5_data2_Hom_L'] = 0

algorithms['gap=.1_data2_Hom_D'] = 0
algorithms['gap=.2_data2_Hom_D'] = 0
algorithms['gap=.3_data2_Hom_D'] = 0
algorithms['gap=.4_data2_Hom_D'] = 0
algorithms['gap=.5_data2_Hom_D'] = 0

# for alg_name in algorithms.keys():
# 	if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
# 		SampleComList_L_Hom[alg_name] = []
# 		CommCostList_L[alg_name] = []
# 	if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
# 		SampleComList_U_Hom[alg_name] = []
# 		CommCostList_U[alg_name] = []
# 	if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
# 		SampleComList_D_Hom[alg_name] = []
# 		CommCostList_D[alg_name] = []
# 	print(alg_name)
# exit(0)

with open('/nfs/stak/users/songchen/research/Async-LinUCB/MovLinear/AccCommCost_dataset0_05_23_01_16.csv', 'r') as f:
	num = 0
	for line in f:
		if num == 0:
			num += 1
			continue
		elif num <= 5:
			num += 1
			continue
		elif num <= 10:
			CommCostList_L.append(int(line))
			num += 1
		else:
			CommCostList_D.append(int(line))
			num += 1

with open('/nfs/stak/users/songchen/research/Async-LinUCB/MovLinear/SampleComplex_dataset0_05_23_01_16.csv', 'r') as f:
	num = 0
	for line in f:
		if num == 0:
			num += 1
			continue
		elif num <= 5:
			num += 1
			SampleComList_U_Hom.append(int(line))
		elif num <= 10:
			SampleComList_L_Hom.append(int(line))
			num += 1
		else:
			SampleComList_D_Hom.append(int(line))
			num += 1


# print(CommCostList_L)

# exit(0)
# # plot the results
fig, axa = plt.subplots(2, 1, sharex='all')
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.2)

sx = []
# syl = []
# syu = []
sylh = []
syuh = []
sydh = []

print("=====Sample Complexity=====")
# for alg_name in algorithms.keys():
# 	# if alg_name[17] == 'U' and alg_name[13:16] == 'Sin':
# 	# 	syu.append(SampleComList_U[alg_name])
# 	# 	sx.append(float(alg_name[5])/10)
# 	# 	print('%s: %.2f' % (alg_name,SampleComList_U[alg_name][-1]))
# 	# if alg_name[17] == 'L' and alg_name[13:16] == 'Sin':
# 	# 	syl.append(SampleComList_L[alg_name])
# 	# 	print('%s: %.2f' % (alg_name, SampleComList_L[alg_name][-1]))

		
# 		sylh.append(SampleComList_L_Hom)
# 		# print('%s: %.2f' % (alg_name, SampleComList_L_Hom[alg_name][-1]))
# 	if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
# 		sx.append(float(alg_name[5])/10)
# 		syuh.append(SampleComList_U_Hom)
# 		# print('%s: %.2f' % (alg_name, SampleComList_U_Hom[alg_name][-1]))
# 	if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
# 		sydh.append(SampleComList_D_Hom)
		# print('%s: %.2f' % (alg_name, SampleComList_D_Hom[alg_name][-1]))

sx = [0.1, 0.2, 0.3, 0.4, 0.5]
sylh = copy.deepcopy(SampleComList_L_Hom)
syuh = copy.deepcopy(SampleComList_U_Hom)
sydh = copy.deepcopy(SampleComList_D_Hom)

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

cx = [0.1, 0.2, 0.3, 0.4, 0.5]
cyl = copy.deepcopy(CommCostList_L)
cyd = copy.deepcopy(CommCostList_D)

print("=====Comm Cost=====")
# for alg_name in algorithms.keys():
# 	if alg_name[17] == 'L' and alg_name[13:16] == 'Hom':
# 		cx.append(float(alg_name[5])/10)
# 		cyl.append(CommCostList_L)
# 		# print('%s: %.2f' % (alg_name, CommCostList_L[alg_name][-1]))
# 	if alg_name[17] == 'U' and alg_name[13:16] == 'Hom':
# 		cyu.append(CommCostList_U)
# 		# print('%s: %.2f' % (alg_name, CommCostList_U[alg_name][-1]))
# 	if alg_name[17] == 'D' and alg_name[13:16] == 'Hom':
# 		cyd.append(CommCostList_D)
		# print('%s: %.2f' % (alg_name, CommCostList_D[alg_name][-1]))
axa[1].plot(cx, cyl, marker='v', linestyle='dotted', label='LinGapE-Sync')
# axa[1].plot(cx, cyu, marker='o', linestyle='dotted', label='LinGapE-Single')
axa[1].plot(cx, cyd, marker='s', linestyle='dotted', label='FedALinPE')
axa[1].set_xlabel("Expected Reward Gap")
axa[1].set_ylabel("Communication Cost")
axa[1].legend(loc='upper right')
# plt.savefig(os.path.join(save_Linear_Case, "SamConAndCommCost_dataset" + self.dataset + str(timeRun) + '.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('test.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()