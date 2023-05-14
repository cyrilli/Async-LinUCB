import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint
import json

class Article():	
	def __init__(self, aid, FV=None):
		self.id = aid
		self.featureVector = FV
		

class ArticleManager():
	def __init__(self, dimension, n_articles, argv, theta):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+AG"
		self.theta = theta

	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].featureVector.tolist())) + '\n')

	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, featureVector = json.loads(line)
				articles.append(Article(aid, np.array(featureVector)))
		return articles

	def GenerateArticles(self):
		# feature_matrix = np.empty([self.n_articles, self.dimension])
		# for i in range(self.dimension):
		# 	print(i)
		# 	feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0*(self.dimension-i)/self.dimension), self.n_articles)
		# 	print(feature_matrix[:,i])
		# 	print()
		# 	# exit(0)
		# # exit(0)
		# for i in range(self.n_articles):
		# 	print(feature_matrix[i])
		# print()
		# print()
		# for key in range(self.n_articles):
		# 	featureVector = feature_matrix[key]
		# 	print(featureVector)
		# 	l2_norm = np.linalg.norm(featureVector, ord =2)
		# 	# print(featureVector)
		# 	articles.append(Article(key, featureVector/l2_norm ))
		target_reward = [0.1,0.2,0.3,0.4,0.5]
		for target in target_reward:
			print("generate data for true gap close to: ", target)
			articles = []
			flag = True
			n = 0
			max_reward = -100
			second_reward = -100
			while(flag):
				featureVector = np.empty([self.dimension])
				for i in range(self.dimension):
					featureVector[i] = np.random.normal(0, np.sqrt(1.0*(self.dimension-i)/self.dimension), 1)
				l2_norm = np.linalg.norm(featureVector, ord =2)
				FV = featureVector/l2_norm
				reward = FV.dot(self.theta)
				if n == self.n_articles:
					flag = False
				elif n == 0 and reward >= 0.7:
					articles.append(Article(n, FV))
					n += 1
					max_reward = reward
				elif n == 1 and max_reward - reward >= (target-0.0005) and max_reward - reward <= (target+0.0005):
					articles.append(Article(n, FV))
					n += 1
					second_reward = reward
				elif n > 1 and reward < second_reward:
					articles.append(Article(n, FV))
					n += 1
			filename = 'ArticlesForHomo_' + str(target) + '_' + str(self.n_articles) + '.dat'
			self.saveArticles(articles, filename)
			# tmp = np.zeros(self.n_articles, dtype=float)
			# print()
			# for i in range(self.n_articles):
			# 	print(articles[i].featureVector)
				# tmp[i] = self.theta.dot(articles[i].featureVector)
			# tmp = np.sort(tmp)
			# gap = tmp[self.n_articles-1] - tmp[self.n_articles-2]
			# print(tmp)
			# print(gap)
			# exit(0)

		# for a in articles:
		# 	print(a.featureVector)

