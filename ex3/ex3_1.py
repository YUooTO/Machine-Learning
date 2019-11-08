import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex3Linx.dat").reshape((-1, 1))
y = np.loadtxt("ex3Liny.dat").reshape((-1, 1))

plt.scatter(x, y, marker='+')
plt.show()
plt.scatter(x, y, marker='.')
# plt.show()


def solve(x, theta):
    tot = len(x)
    y = []
    for i in range(tot):
        xx = []
        xx.append(1)
        xx.append(x[i])
        for j in range(2, 6):
            xx.append(x[i] ** j)
        xx = np.mat(xx)
        tmp = xx * theta
        y.append(tmp[0, 0])
    return y

# 预处理：


x = np.hstack((np.ones([7, 1]), x))
for i in range(2, 6):
    x_ = np.empty([7, 1])
    for j in range(7):
        x_[j] = x[j][1] ** i
    x = np.hstack((x, x_))

x = np.mat(x)
y = np.mat(y)

z = np.zeros([6, 6])
for i in range(1, 6):
    z[i, i] = 1

rate = 1
theta = np.mat((x.T * x + rate * z).I * x.T * y)
x1 = np.linspace(-1, 1, 100)
y1 = solve(x1, theta)
y1 = np.asarray(y1)

plt.plot(x1, y1, c='r')
plt.show()
