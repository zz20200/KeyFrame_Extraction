import cv2
import os
import matplotlib.pyplot as plt
from skimage import filters
import numpy as np
import matplotlib.ticker as ticker

P=[7, 14, 25, 37, 41, 49, 55, 62, 66, 71]
Q = [P[i] - P[i - 1] for i in range(1, len(P))]
m = np.mean(Q)
density = []  # 对于每个index元素的点密度矩阵
for i in range(0, len(P)):
    c = 0
    for x in P:
        # 以每个元素为原点，dis_mean为半径的区间内，计算index元素的个数c
        if (x < (P[i] + m)) & (x > (P[i] - m)):
            c += 1
    density.append(c)
print(density)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(1,1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.scatter(P,density,s=45,c='r',marker='o')

density_mean = np.mean(density)
print(density_mean)
x=np.linspace(0,80,500)
y = [density_mean for i in x]
plt.plot(x,y,linestyle='--',color='black',linewidth=3)

y = np.linspace(0,5,100)
x=[2 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[7 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[12 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[17 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[22 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[27 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[32 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[37 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[42 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[47 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[52 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)


y = np.linspace(0,5,100)
x=[57 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[62 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[67 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[72 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

y = np.linspace(0,5,100)
x=[77 for i in y]
plt.plot(x,y,linestyle='--',color='black',linewidth=1)

plt.text(12, 3.3,'T=2.29',color='black',fontsize=15)
plt.annotate(s='', xy=(9, 2.29),xytext=(15, 3.2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.scatter(41, 3, color='', marker='o', edgecolors='purple', s=200)
plt.scatter(44, 3, color='', marker='o', edgecolors='purple', s=200)
plt.scatter(66, 3, color='', marker='o', edgecolors='purple', s=200)
plt.scatter(71, 4, color='', marker='o', edgecolors='purple', s=200)
plt.scatter(74, 3, color='', marker='o', edgecolors='purple', s=200)

# plt.grid(axis="y")
plt.xlim(int(0),80 )
plt.ylim(int(0),5,5)

plt.show()