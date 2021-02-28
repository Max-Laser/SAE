from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats
import time
# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 5#25
maxIter = 25#100
numTrials = 5#10

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
rew_step=np.zeros((maxIter,))
rew_mean=np.zeros((numTrials,maxIter))
rew_plot=np.zeros((maxIter,))
error=np.zeros((maxIter,))
alpha=0.1

#To plot
x_achse=np.zeros((maxIter))

for k in range(0,numTrials):
    mu = np.zeros((numDim,))

    for i in range(0,maxIter):
        grad_sample = np.zeros((numSamples,numDim))

        for t in range(0,numSamples):
            theta= np.random.multivariate_normal(mu, Sigma)
            rew[t]= env.getReward(theta)

        rew_step =np.mean(rew)
        print(i)

        for t in range(0, numSamples):
            #grad_sample[t,:] =  np.dot(np.linalg.inv(Sigma), (theta - mu)) * (rew[t])
            grad_sample[t, :] = np.dot(np.linalg.inv(Sigma), (theta - mu)) * (rew[t]-rew_step)

        grad_sample_mean= np.mean(grad_sample)

        mu += alpha * grad_sample_mean

        rew_mean[k,:]= rew_step


for i in range(0,maxIter):
    h=rew_mean[:, i]
    rew_plot[i],error[i]= mean_confidence_interval(data=rew_mean[:,i])
    x_achse[i] = i




plt.plot(x_achse,rew_plot,'ob',label='average return')
plt.errorbar(x_achse,rew_plot ,yerr = error,fmt ='o')
plt.legend()
plt.show()

for i in range(numSamples):
    if useBaseline:
        rewardGradients[i,:] = (np.linalg.inv(sigmaMatrix) @ (theta_samples[i,:]-mu)) * (rewards[i] - averageReward)
    else
        rewardGradients[i,:] = (np.linalg.inv(sigmaMatrix) @ (theta_samples[i,:]-mu)) * rewards[i]




