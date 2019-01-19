import numpy as np
import matplotlib.pyplot as plt
import math
import random

a1=np.array(np.arange(1,50,2))
print(a1)
a2=np.array(np.sin(a1))
a3=np.array(np.cos(a1))
a4=np.array(np.random.normal(0,0.01,size=25))
print('a2',a2)
print('a3',a3)
print('a4',a4)

#使用折线图进行可视化
y=np.multiply(a2,a2.T)
z=np.multiply(a2,a3.T)
w=np.multiply(a2,a4.T)

plt.plot(a1,a2)
plt.plot(a1,y)
plt.plot(a1,z)
plt.plot(a1,w)
plt.show()



