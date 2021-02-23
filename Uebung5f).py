import numpy as np
import matplotlib.pyplot as plt





q1 = np.loadtxt("q1.txt")
q2 = np.loadtxt("q2.txt")
q3 = np.loadtxt("q3.txt")
q1Abl = np.loadtxt("q1Abl.txt")
q2Abl = np.loadtxt("q2Abl.txt")
q3Abl = np.loadtxt("q3Abl.txt")
q1AblAbl = np.loadtxt("q1AblAbl.txt")
q2AblAbl = np.loadtxt("q2AblAbl.txt")
q3AblAbl = np.loadtxt("q3AblAbl.txt")
u1 = np.loadtxt("u1.txt")
u2 = np.loadtxt("u2.txt")
u3 = np.loadtxt("u3.txt")

g = 9.81
m2Zae = 0
m2Ne = 0
m1Zae = 0
m1Ne = 0
for i in range(0, 100):
    m2Zae = m2Zae + u3[i]*(q3AblAbl[i] - q3[i]*(q2Abl[i]**2))
    m2Ne = m2Ne +(q3AblAbl[i]-q3[i]*(q2Abl[i]**2))*(q3AblAbl[i]-q3[i]*(q2Abl[i]**2))
    m1Zae = m1Zae + u1[i]*(q1AblAbl[i]+g)
    m1Ne = m1Ne + (q1AblAbl[i] + g)**2

m2 = (m2Zae/(m2Ne))
m1 = ((m1Zae/m1Ne) - m2)

u1approx = np.zeros((np.shape(u1)))
u2approx = np.zeros((np.shape(u2)))
u3approx = np.zeros((np.shape(u3)))
t = np.zeros((np.shape(u1)))

for i in range(0, 100):
    u1approx[i] = (m1 + m2)*(q1AblAbl[i]+g)
    u2approx[i] = m2*(2*q3Abl[i]*q2Abl[i]*q3[i]+(q3[i]**2)*q2AblAbl[i])
    u3approx[i] = m2*(q3AblAbl[i]-q3[i]*(q2Abl[i]**2))
    if i>0:
        t[i] = t[i-1]+ 0.002
    else:
        t[i] = 0

#names = ['Desired_1', 'Desired_2']
#plt.legend(tuple(names))
#plt.xlabel('time[s]', fontsize=15)
#plt.ylabel('u1[N]', fontsize=15)

plt.plot(t,u1,label='u1')
plt.plot(t,u1approx, label='u1approx')
plt.legend()
plt.show()
plt.plot(t,u2,label='u2')
plt.plot(t,u2approx, label='u2approx')
plt.legend()
plt.show()
plt.plot(t,u3,label='u3')
plt.plot(t,u3approx, label='u3approx')
plt.legend()
plt.show()
