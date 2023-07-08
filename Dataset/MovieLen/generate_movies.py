import argparse
import sys
import numpy as np
sys.path.append("/nfs/stak/users/songchen/research/Async-LinUCB")
from util_functions import gaussianFeature

from Movies import MovieManager

def loadtheta(filename):
	theta = []
	with open(filename, 'r') as f:
		for line in f:
			theta.append(float(line))
	return theta


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description= ' ')
	parser.add_argument('--N', dest='N', help='total number of movies(K)')
	args = parser.parse_args()

	if args.N:
		n_movies = int(args.N)
	else:
		n_movies = 10
	
	dimension = 25
	theta_file = '150wtheta.dat'

	theta = loadtheta(theta_file)
	MM = MovieManager(dimension, n_movies, argv={'l2_limit': 1}, theta=theta)
	MM.GenerateMovies()