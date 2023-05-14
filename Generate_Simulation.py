import argparse
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
	users = UM.simulateThetaForHomoUsers()
	for i in range(len(users)):
		print(users[i].theta)
	exit(0)
	AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
	articles = AM.simulateArticlePool()