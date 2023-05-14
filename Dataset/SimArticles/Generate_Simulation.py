import argparse
import sys
import numpy as np
sys.path.append("/nfs/stak/users/songchen/research/Async-LinUCB")
from util_functions import gaussianFeature

from Articles import ArticleManager
from Users import UserManager


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description= ' ')
	parser.add_argument('--N', dest='N', help='total number of clients')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	args = parser.parse_args()

	if args.contextdim:
		context_dimension = int(args.contextdim)
	else:
		context_dimension = 5
	if args.N:
		n_users = int(args.N)
	else:
		n_users = 100
	n_articles = 10

	UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
	
	# Since in this case we only consider the Homogeneous case
	# So all the vector features of user is the same
	# Generate user data at first and then save to the file 'users.dat'
	# users = UM.simulateThetaForHomoUsers()
	# UM.saveHomoUsers(users, 'usersHomo.dat')
	
	# load the saved user data
	users = UM.loadHomoUsers('usersHomo.dat')
	user_feature_vector = users[0].theta
	AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1}, theta=user_feature_vector)
	# AM.GenerateArticles()
	gap = 0.5
	articles = AM.loadArticles('ArticlesForHomo_' + str(gap) + '_' + str(n_articles) + '.dat')
	
	# test reward:
	reward = np.zeros(n_articles, dtype=float)
	for i in range(n_articles):
		reward[i] = user_feature_vector.dot(articles[i].featureVector)
	reward = np.sort(reward)
	expected_gap = reward[n_articles-1] - reward[n_articles-2]
	print(reward)
	print()
	print(expected_gap)
	