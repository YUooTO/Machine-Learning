import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# 数据处理
x = np.loadtxt("ex1_1x.dat").reshape((-1, 1))
y = np.loadtxt("ex1_1y.dat")
tot = x.shape[0]
a = np.ones([tot, 1])
x = np.hstack((a, x))

# 训练模型

theta = [0, 0]
iterations = 1500
rate = 0.07

for it in range(iterations):
    for i in range(len(theta)):
        z = 0.0
        for j in range(tot):
            w = np.dot(x[j], theta)
            w -= y[j]
            w *= x[j][i]
            z += w
        z /= tot
        theta[i] -= rate * z
print(theta)
print(np.dot(theta, [1, 3.5]))
print(np.dot(theta, [1, 7]))
plt.scatter([b[1] for b in x], y)
plt.plot([0, 9], [theta[0], 9*theta[1] + theta[0]])
plt.show()

# 下面是三维画图


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
            val += (X[i][j] + Y[i][j] * x[k][1] - y[k])**2
        val /= tot * 2
        list1.append(val)
    list0.append(list1)
Z = np.array(list0)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()



