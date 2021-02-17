import numpy as np

m2=0
for i in range (0,99):
    m2 = 5#u3[i]/(q3AblAbl[i]-q3[i](q2Abl[i]**2))+m2

for i in range (0,99):
    m2 = u1[i]/(q3AblAbl[i]-q3[i](q2Abl[i]**2))+m2

np.zeros((np.shape()))