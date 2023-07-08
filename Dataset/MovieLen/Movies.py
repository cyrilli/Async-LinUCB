import numpy as np
import pandas as pd
import json

class Movie():
	def __init__(self, mid, FV=None):
		self.mid = mid
		self.featureVector = FV
	
class MovieManager():
	def __init__(self, dimension, n_movies, argv, theta):
		self.dimension = dimension
		self.n_movies = n_movies
		self.argv = argv
		self.theta = theta
		
		self.data_file = '150wdata.csv'

	def saveMovies(self, movies, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(movies)):
				f.write(json.dumps((movies[i].mid, movies[i].featureVector.tolist())) + '\n')

	def loadMovies(self, filename):
		movies = []
		with open(filename, 'r') as f:
			for line in f:
				mid, featureVector = json.loads(line)
				movies.append(Movie(mid, np.array(featureVector)))
		return movies

	def GenerateMovies(self):
		target_reward = [0.1, 0.2, 0.3, 0.4, 0.5]
		# target_reward = [0.5]

		data = pd.read_csv(self.data_file)
		df = pd.DataFrame(data)
		data = data.drop(columns='ArticleID')
		data = data.drop(columns='Unnamed: 0')
		# print(data)
		# for index, row in data.iterrows():
		# 	print(index, row.values.tolist())
		# print(data.shape)
		# exit(0)

		all_reward = np.zeros(data.shape[0])
		all_FV = np.zeros((data.shape[0], data.shape[1]))

		for index, row in data.iterrows():
			featuerVector = np.array(row.values.tolist())
			reward = featuerVector.dot(self.theta)
			all_FV[index] = featuerVector
			all_reward[index] = reward

		# print(np.sort(np.max(all_reward) - all_reward)[0:100])
		# print(np.sort(all_reward)[-100:-1])
		# print(all_FV[np.argmax(all_reward)])
		# exit(0)

		for target in target_reward:
			print("generate data for true gap close to: ", target)
			movies = []
			flag = False
			n = 0
			max_reward = -100
			# max_reward = np.max(all_reward)
			# movies.append(Movie(n, all_FV[np.argmax(all_reward)]))
			# n += 1
			for i in range(len(all_reward)):
				if n == self.n_movies:
					flag = True
				if flag:
					break
				if n == 0 and all_reward[i] >= 0.36 and all_reward[i] <= 0.39:
					movies.append(Movie(n, all_FV[i]))
					max_reward = all_reward[i]
					i = 0
					n += 1
				elif n > 0 and max_reward - all_reward[i] >= (target-0.01) and max_reward - all_reward[i] <= (target+0.01):
					movies.append(Movie(n, all_FV[i]))
					n += 1
			tmp = np.zeros(self.n_movies, dtype=float)
			print(n)
			print()
			for i in range(self.n_movies):
				# print(movies[i].featureVector)
				tmp[i] = movies[i].featureVector.dot(self.theta)
			tmp = np.sort(tmp)
			gap = tmp[self.n_movies-1] - tmp[self.n_movies-2]
			print(tmp)
			print(gap)
			filename = 'MovieLen' + str(target) + '_' + str(self.n_movies) + '.dat'
			self.saveMovies(movies, filename)
			# exit(0)