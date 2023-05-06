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

from lib.AsyncLinUCB_AM import AsyncLinUCB_AM

class simulateOnlineData(object):
    def __init__(self, context_dimension, testing_iterations, plot, articles,
                 users, noise=lambda: 0, signature='', NoiseScale=0.0, poolArticleSize=None, name_label=None):

        self.simulation_signature = signature
        self.name_label = name_label
        self.context_dimension = context_dimension
        self.testing_iterations = testing_iterations
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

    def getTheta(self):
        Theta = np.zeros(shape=(self.context_dimension, len(self.users)))
        for i in range(len(self.users)):
            Theta.T[i] = self.users[i].theta
        return Theta

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)

    def getReward(self, user, pickedArticle):
        return np.dot(user.theta, pickedArticle.featureVector)

    def GetOptimalReward(self, user, articlePool):
        maxReward = float('-inf')
        maxx = None
        for x in articlePool:
            reward = self.getReward(user, x)
            if reward > maxReward:
                maxReward = reward
                maxx = x
        return maxReward, maxx

    def getL2Diff(self, x, y):
        return np.linalg.norm(x - y)  # L2 norm

    def regulateArticlePool(self):
        # Randomly generate articles
        self.articlePool = sample(self.articles, self.poolArticleSize)

    def runAlgorithms(self, algorithms):
        self.startTime = datetime.datetime.now()
        timeRun = self.startTime.strftime('_%m_%d_%H_%M')
        filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + self.name_label + '.csv')
        filenameWriteCommCost = os.path.join(save_address, 'AccCommCost' + timeRun + self.name_label + '.csv')
        filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + self.name_label + '.csv')

        tim_ = []
        BatchCumlateRegret = {}
        CommCostList = {}
        AlgRegret = {}
        ThetaDiffList = {}
        ThetaDiff = {}

        # Initialization
        # userSize = len(self.users)
        for alg_name, alg in algorithms.items():
            AlgRegret[alg_name] = []
            CommCostList[alg_name] = []
            BatchCumlateRegret[alg_name] = []
            if alg.CanEstimateUserPreference:
                ThetaDiffList[alg_name] = []

        with open(filenameWriteRegret, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write('\n')

        with open(filenameWriteCommCost, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write('\n')

        with open(filenameWritePara, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) + 'Theta' for alg_name in ThetaDiffList.keys()]))
            f.write('\n')

        for iter_ in range(self.testing_iterations):
            # prepare to record theta estimation error
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] = 0

            # for u in self.users:
            u = random.choices(population=self.users, weights=None, k=1)[0]
            self.regulateArticlePool()
            noise = self.noise()
            # get optimal reward for user x at time t
            OptimalReward, OptimalArticle = self.GetOptimalReward(u, self.articlePool)
            OptimalReward += noise

            for alg_name, alg in algorithms.items():
                pickedArticle = alg.decide(self.articlePool, u.id)
                reward = self.getReward(u, pickedArticle) + noise
                alg.updateParameters(pickedArticle, reward, u.id)

                regret = OptimalReward - reward  # pseudo regret, since noise is canceled out
                AlgRegret[alg_name].append(regret)
                CommCostList[alg_name].append(alg.totalCommCost)

                # update parameter estimation record
                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))

            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiffList[alg_name] += [ThetaDiff[alg_name]]

            if iter_ % self.batchSize == 0:
                self.batchRecord(iter_)
                tim_.append(iter_)
                for alg_name in algorithms.keys():
                    BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))

                with open(filenameWriteRegret, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')
                with open(filenameWriteCommCost, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')
                with open(filenameWritePara, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
                    f.write('\n')

        if (self.plot == True):  # only plot
            # # plot the results
            fig, axa = plt.subplots(2, 1, sharex='all')
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)

            print("=====Regret=====")
            for alg_name in algorithms.keys():
                axa[0].plot(tim_, BatchCumlateRegret[alg_name], label=alg_name)
                print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
            axa[0].legend(loc='upper left', prop={'size': 9})
            axa[0].set_xlabel("Iteration")
            axa[0].set_ylabel("Accumulative Regret")
            axa[1].set_ylim(bottom=0, top=200)

            print("=====Comm Cost=====")
            for alg_name in algorithms.keys():
                axa[1].plot(tim_, CommCostList[alg_name], label=alg_name)
                print('%s: %.2f' % (alg_name, CommCostList[alg_name][-1]))
            axa[1].set_xlabel("Iteration")
            axa[1].set_ylabel("Communication Cost")
            axa[1].set_ylim(bottom=0, top=20000)

            plt.savefig(os.path.join(save_address, "regretAndcommCost" + "_" + str(timeRun) + self.name_label + '.png'),
                        dpi=300,
                        bbox_inches='tight', pad_inches=0.0)
            plt.show()

        finalRegret = {}
        for alg_name in algorithms.keys():
            finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
        return finalRegret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--T', dest='T', help='total number of iterations')
    parser.add_argument('--N', dest='N', help='total number of clients')
    parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
    parser.add_argument('--globaldim', dest='global_dim', help='Set dimension of globally shared context features.')
    parser.add_argument('--accuracy', dest = 'E', type=float, help = 'Set the target accuracy')
    parser.add_argument('--confidence', dest = 'C', type=float, help = 'Set the confidence level')
    args = parser.parse_args()

    ## Environment Settings ##
    if args.contextdim:
        context_dimension = int(args.contextdim)
    else:
        context_dimension = 5
    if args.T:
        testing_iterations = int(args.T)
    else:
        testing_iterations = 5000
    if args.N:
        n_users = int(args.N)
    else:
        n_users = 1000
    if args.globaldim:
        global_dim = int(args.globaldim)
        assert global_dim < context_dimension
    else:
        global_dim = 5 # change the default global dimention to 5
	# Set the target accuracy(epsilon)
    if args.E:
        epsilon = float(args.E)
    else:
        epsilon = 2*(1-np.cos(0.01))
    # Set the target confidence level(delta)
    if args.C:
        confidence = float(args.C)
    else:
        confidence = 0.05
    NoiseScale = 0.1  # standard deviation of Gaussian noise
    n_articles = 1000
    poolArticleSize = 10

    local_dim = context_dimension - global_dim

    name_label = "T{}N{}dg{}".format(testing_iterations, n_users, global_dim)

    ## Set Up Simulation ##
    UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    users = UM.simulateThetaForHeteroUsers(global_dim=global_dim)
    AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool()

    simExperiment = simulateOnlineData(context_dimension=context_dimension,
                                       testing_iterations=testing_iterations,
                                       plot=True,
                                       articles=articles,
                                       users=users,
                                       noise=lambda: np.random.normal(scale=NoiseScale),
                                       signature=AM.signature,
                                       NoiseScale=NoiseScale,
                                       poolArticleSize=poolArticleSize,
                                       name_label=name_label)

    ## Initiate Bandit Algorithms ##
    algorithms = {}

    lambda_ = 0.1
    delta = 1e-1

    algorithms['gamma=5'] = AsyncLinUCB_AM(dimension_g=global_dim, dimension_l=local_dim, alpha=-1,
                                           lambda_=lambda_,
                                           delta_=delta,
                                           NoiseScale=NoiseScale, gammaU=5, gammaD=5)


    ## Run Simulation ##
    print("Starting for ", simExperiment.simulation_signature)
    simExperiment.runAlgorithms(algorithms)