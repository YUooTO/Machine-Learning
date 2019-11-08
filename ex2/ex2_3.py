import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex2x.dat")
y = np.loadtxt("ex2y.dat").reshape((-1, 1))

tot = x.shape[0]
x = np.hstack((np.ones([80, 1]), x))
x = np.mat(x)
x_pos = x[:40, :]
x_neg = x[40:, :]

plt.scatter([b[0, 1] for b in x_pos], [b[0, 2] for b in x_pos], marker='+')
plt.scatter([b[0, 1] for b in x_neg], [b[0, 2] for b in x_neg], marker='o')


def g(k):
    return 1/(1 + np.exp(-k))


theta = np.zeros([3, 1])
e = 1e-6
pre = 0
it = 0

L = []

while 1:
    now = 0
    it = it + 1
    h = np.mat(np.zeros([3, 3]))
    w = np.mat(np.zeros([1, 3]))
    for i in range(tot):
        tmp = g(np.dot(x[i], theta))
        now = now + (-y[i]*np.log(tmp) - (1 - y[i])*np.log(1 - tmp))
        z = (tmp - y[i]) * x[i]
        w += z
        h += (tmp[0, 0]*(1-tmp[0, 0]) * x[i].T * x[i])
    w /= tot
    now /= tot
    h /= tot
    L.append(now[0, 0])
    if np.abs(now - pre) <= e:
        break
    pre = now
    theta = theta - h.I * w.T

print(it)
print(theta)
x1 = np.linspace(10, 70, 70)
y = [xx*(-1*theta[1, 0]/theta[2, 0]) - theta[0, 0]/theta[2, 0] for xx in x1]
plt.plot(x1, y)
plt.show()
X = [1, 20, 80]
X = np.mat(X)
print(1 - g(np.dot(X, theta))[0, 0])
L = np.asarray(L)
plt.plot(L)
plt.show()
