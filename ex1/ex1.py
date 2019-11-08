import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#数据处理
fp = open("ex1_1x.dat", "r")
fq = open("ex1_1y.dat", "r")
x = []
y = []
for i in fp.readlines():
	x.append([1, float(i)])
for i in fq.readlines():
	y.append(float(i))
print(y)
tot = len(x)
#训练模型
theta = [0, 0]
iterations = 1500
rate = 0.07
for itera in range(iterations):
	for i in range(len(theta)):
		z = 0.0
		for j in range(tot):
			w = 0.0
			for k in range(len(x[j])):
				w += x[j][k] * theta[k]
			w -= y[j]
			w *= x[j][i]
			z += w
		z /= tot
		theta[i] -= rate * z;
#可视化模型
# print(theta)

plt.scatter([a[1] for a in x], y)
plt.plot([0, 9], [theta[0], 9 * theta[1] + theta[0]])
plt.show()

#下面为三维画图

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-3, 3, 0.06)
Y = np.arange(-1, 1, 0.02)
X, Y = np.meshgrid(X, Y)
list0 = []
[rows, cols] = X.shape

for i in range(rows):
    list1 = []
    for j in range(cols):
    	val = 0
    	for k in range(tot):
    		val += (X[i][j] + Y[i][j] * x[k][1] - y[k]) * (X[i][j] + Y[i][j] * x[k][1] - y[k])
    	val /= tot * 2
    	list1.append(val)
    list0.append(list1)

Z = np.array(list0)
print(Z.shape)
plt.savefig('fig.png', bbox_inches='tight')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
