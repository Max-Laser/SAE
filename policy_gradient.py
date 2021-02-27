from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats
import time
# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials =10

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m,h

# YOUR CODE HERE

grad_sample = np.zeros((numSamples,))
mu=np.zeros((numDim,))
Sigma=np.diag(np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))#TODO evtl 100
rew = np.zeros((numSamples,))
rew_step=np.zeros((maxIter,))
rew_mean=np.zeros((numTrials,))

#To plot
x_achse=np.zeros((numTrials))
error=np.zeros((numTrials,))

for k in range(0,numTrials):
    for i in range(0,maxIter):
        for t in range(0,numSamples):
            theta= np.random.multivariate_normal(mu, Sigma)
            rew[t]= env.getReward(theta)
            grad_sample[t] = (1 / maxIter) * np.linalg.inv(Sigma) * (theta - mu[i]) * (rew[t] - 0.01)
        #theta_mean = np.mean(theta)
        #rew_mean= np.mean(rew)
        grad= np.mean(grad_sample)
        rew_mean[i] = np.mean(rew)
        p=mu[i]+0.1*grad
        mu[i+1]=p
        """""
        temp = np.zeros((5, 1))
        for h in range(0, 5):
            temp[h] = theta
        print(env.getReward(temp))
        """""
        print(i)

    x_achse[k]=k
    rew_mean[k],error[k]= mean_confidence_interval(data=mu)


plt.plot(x_achse,np.transpose(rew_mean1),'ob',label='average return')
plt.errorbar(x_achse, np.transpose(rew_mean1) ,yerr = error,fmt ='o')
plt.legend()
plt.show()






