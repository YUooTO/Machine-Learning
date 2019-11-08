import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex1_2x.dat")
y = np.loadtxt("ex1_2y.dat")
tot = x.shape[0]
x = np.hstack((np.ones([tot, 1]), x))
x = np.vstack((x, [1, 1650, 3]))
sigma = np.std(x, 0)

mu = np.mean(x, 0)
for i in range(x.shape[0]):
    x[i][1] = (x[i][1] - mu[1]) / sigma[1]
    x[i][2] = (x[i][2] - mu[2]) / sigma[2]

X = x[tot]
x = np.resize(x, (tot, 3))
rate = [0.05, 0.15, 1]
iterations = 50
G = []
for ra in range(3):
    theta = np.zeros(x.shape[1])
    J = np.zeros([iterations, 1])
    for it in range(iterations):
        now = np.zeros(len(theta))
        for i in range(tot):
            z = np.dot(theta, x[i]) - y[i]
            now += z * x[i]
        now /= tot
        theta -= rate[ra] * now
        c = np.dot(x, theta) - y
        d = c
        J[it] = np.dot(c, d) / (2 * tot)
    if ra == 1:
        value = np.dot(theta, X)
    G.append(J)
print(theta)
print(value)
e = np.arange(50)
plt.plot(e, G[0], 'b', label='0.05')
plt.plot(e, G[1], 'r', label='0.15')
plt.plot(e, G[2], 'y', label='1')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("J")
plt.show()
