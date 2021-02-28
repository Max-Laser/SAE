from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats
import math
import time
# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m,h

# YOUR CODE HERE

grad_sample = np.zeros((numSamples,))

Sigma=np.diag(np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))#TODO evtl 100
rew = np.zeros((numSamples,))
rew_step=0#=np.zeros((maxIter,))
rew_mean=np.zeros((numTrials,maxIter))
rew_plot=np.zeros((maxIter,))
error=np.zeros((maxIter,))
alpha=0.1

#To plot
x_achse=np.zeros((maxIter))



for k in range(0,numTrials):
    mu = np.zeros((numDim,))
    for i in range(0,maxIter):
        grad_sample = np.zeros((numDim,))
        theta = np.zeros((numSamples, numDim))
        for t in range(0,numSamples):
            theta[t,:]= np.random.multivariate_normal(mu, Sigma)
            rew[t]= env.getReward(theta[t])
            k=0

        rew_step =np.mean(rew)
        rew_mean[k, i] = rew_step
        print(i)
        rew_grad=np.zeros((numSamples,numDim))
        for t in range(0,numSamples):
            #rew_grad[t][:]= np.dot(np.linalg.inv(Sigma), (theta[t] - mu) * (rew[t]))
            #print(rew[t])
            rew_grad[t,:] = np.dot(np.linalg.inv(Sigma), (theta[t] - mu)) * ((rew[t])- rew_step)
            #test=theta[t]


        grad_sample_mean= rew_grad.mean(axis=0)

        mu =mu+ alpha * grad_sample_mean

