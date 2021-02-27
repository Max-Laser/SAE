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
maxIter = 10#100
numTrials = 3#10

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

# YOUR CODE HERE
theta = np.zeros((numSamples,1))
grad_sample = np.zeros((numSamples,1))
mu=np.zeros((maxIter+1,1))
sigma=10
rew = np.zeros((numSamples,1))
rew_mean=np.zeros((maxIter+1,1))

mu1=np.zeros((numTrials,1))
mu2=np.zeros((numTrials,1))
mu3=np.zeros((numTrials,1))

rew_mean1=np.zeros((numTrials,1))
rew_mean2=np.zeros((numTrials,1))
rew_mean3=np.zeros((numTrials,1))

x_achse=np.zeros((numTrials,1))

for k in range(0,numTrials):
    for i in range(0,maxIter):
        for t in range(0,numSamples):

            theta= np.random.normal(mu[i],10)
            temp=np.zeros((5,1))
            for h in range(0,5):
                temp[h]=theta
            rew[t]= env.getReward(np.transpose(np.transpose(temp)))
            grad_sample[t] = 1 / maxIter * 1 / (100) ** 2 * (theta - mu[i]) * rew[t]
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
    x_achse[k]=k+1
    print(k)
    #mu1[k],mu2[k],mu3[k]= mean_confidence_interval(data=rew_mean)
    rew_mean1[k],rew_mean2[k],rew_mean3[k]= mean_confidence_interval(data=mu)
"""""
plt.plot(x_achse,mu1,label='mean')
plt.plot(x_achse,mu2)
plt.plot(x_achse,mu3)
plt.legend()
plt.show()
"""""
#x_achse=np.array([0,1,2])
#plt.fill_between(x_achse, np.transpose(rew_mean3), np.transpose(rew_mean2), color='c', label='95% confidence')
plt.plot(np.transpose(x_achse),np.transpose(rew_mean1),label='average return')
plt.plot(np.transpose(x_achse),np.transpose(rew_mean2))
plt.plot(np.transpose(x_achse),np.transpose(rew_mean3))
plt.legend()
plt.show()






