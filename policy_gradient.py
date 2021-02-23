from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import time
# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10


# YOUR CODE HERE
theta = np.zeros((numSamples,1))
grad_sample = np.zeros((numSamples,1))
mu=np.zeros((maxIter+1,1))
sigma=10
for k in range(0,numTrials):
    for i in range(0,maxIter):
        for t in range(0,numSamples):
            k= mu[i]
            theta= np.random.normal(mu[i],10)
            temp=np.zeros((5,1))
            for h in range(0,5):
                temp[h]=theta
            rew= env.getReward(np.transpose(np.transpose(temp)))
            grad_sample[t] = 1 / maxIter * 1 / (100) ** 2 * (theta - mu[i]) * rew
        #theta_mean = np.mean(theta)
        #rew_mean= np.mean(rew)
        grad= np.mean(grad_sample)

        p=mu[i]+0.1*grad
        mu[i+1]=p

        temp = np.zeros((5, 1))
        for h in range(0, 5):
            temp[h] = theta
        print(env.getReward(temp))


