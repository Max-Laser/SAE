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
#alpha=0.1
alpha=0.2

#To plot
x_achse=np.zeros((maxIter))

"""""
useBaseline=False
averageRewards= np.zeros((numTrials,maxIter))

for trial in range(0,numTrials):
    mu = np.zeros((numDim,))
    sigma = np.repeat([np.sqrt(10)],numDim)#Todo
    for iterator in range(0,maxIter):
        sigmaMatrix= np.diag(np.power(sigma,2))#Todo
        rewards = np.zeros((numSamples,))#Todo
        theta_samples = np.zeros((numSamples, numDim))
        for t in range(0,numSamples):
            theta_samples[t,:]= np.random.multivariate_normal(mu, sigmaMatrix)
            rewards[t]= env.getReward(theta_samples[t,:])
            k=0

        averageReward= rewards.mean()
        averageRewards[trial,iterator]= averageReward
        rewardGradients=np.zeros((numSamples,numDim))
        print(iterator)
        for i in range(numSamples):
            if useBaseline:
                rewardGradients[i, :] = (np.linalg.inv(sigmaMatrix) @ (theta_samples[i, :] - mu)) * (
                            rewards[i] - averageReward)
            else:
                rewardGradients[i, :] = (np.linalg.inv(sigmaMatrix) @ (theta_samples[i, :] - mu)) * rewards[i]


        averageRewardGradient= rewardGradients.mean(axis=0)
        mu = mu +alpha*averageRewardGradient





for i in range(0,maxIter):
    h=averageRewards[:, i]
    rew_plot[i],error[i]= mean_confidence_interval(data=averageRewards[:,i])
    x_achse[i] = i
"""

for k in range(0,numTrials):
    Sigma = np.diag(np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))  # TODO evtl 100
    mu = np.zeros((numDim,))
    sigmaAlone = np.repeat([np.sqrt(10)],numDim)
    for i in range(0,maxIter):
        Sigma = np.diag(np.array(sigmaAlone[0]**2,sigmaAlone[1]**2, sigmaAlone[2]**2, sigmaAlone[3]**2,sigmaAlone[4]**2, sigmaAlone[5]**2,sigmaAlone[6]**2, sigmaAlone[7]**2, sigmaAlone[8]**2,sigmaAlone[9]**2))  # TODO evtl 1
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
        sigmagrad = np.zeros((numSamples, numDim,numDim))
        for t in range(0,numSamples):
            #rew_grad[t][:]= np.dot(np.linalg.inv(Sigma), (theta[t] - mu) * (rew[t]))
            #print(rew[t])
            rew_grad[t,:] = np.dot(np.linalg.inv(Sigma), (theta[t] - mu)) * ((rew[t])- rew_step)
            SigmaAlone3= np.linalg.inv(np.dot(np.dot(sigmaAlone,sigmaAlone ),sigmaAlone))
            theta_mu= np.power(theta[t]-mu,2)
            sigmagrad[t]= (-1)*np.linalg.inv(sigmaAlone) + np.dot(theta_mu,SigmaAlone3)
            #test=theta[t]

        grad_sample_mean= rew_grad.mean(axis=0)
        grad_sigma_mean= sigmagrad.mean(axis=0)
        sigmaAlone = sigmaAlone + alpha* grad_sigma_mean
        mu =mu+ alpha * grad_sample_mean



for i in range(0,maxIter):
    h=rew_mean[:, i]
    rew_plot[i],error[i]= mean_confidence_interval(data=rew_mean[:,i])
    x_achse[i] = i

plt.plot(x_achse,rew_plot,'ob',label='average return')
plt.errorbar(x_achse,rew_plot ,yerr = error,fmt ='o')
plt.legend()
plt.show()



"""""
for k in range(0,numTrials):
    mu = np.zeros((numDim,))
    Sigma = np.repeat([np.sqrt(10)],numDim)#Todo
    for i in range(0,maxIter):
        SigmaMatrix= np.diag(np.power(Sigma,2))#Todo
        rew = np.zeros((numSamples,))#Todo
        grad_sample = np.zeros((numDim,))
        theta = np.zeros((numSamples, numDim))
        for t in range(0,numSamples):
            theta[t,:]= np.random.multivariate_normal(mu, Sigma)
            rew[t]= env.getReward(theta[t])

        rew_step =np.mean(rew)
        print(i)
        rew_grad=np.zeros((numSamples,numDim))
        for t in range(0,numSamples):
            #rew_grad[t][:]= np.dot(np.linalg.inv(Sigma), (theta[t] - mu) * (rew[t]))
            #print(rew[t])
            rew_grad[t,:] = np.dot(np.linalg.inv(Sigma), (theta[t] - mu) * (((rew[t])- rew_step)))
            #test=theta[t]


        grad_sample_mean= np.mean(rew_grad)

        mu += alpha * grad_sample_mean

        rew_mean[k,i]= rew_step
        
        
for i in range(0,maxIter):
    h=rew_mean[:, i]
    rew_plot[i],error[i]= mean_confidence_interval(data=rew_mean[:,i])
    x_achse[i] = i


"""

