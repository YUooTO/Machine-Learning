import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex2x.dat")
y = np.loadtxt("ex2y.dat").reshape((-1, 1))

tot = x.shape[0]
x = np.hstack((np.ones([80, 1]), x))

x_pos = x[:40, :]
x_neg = x[40:, :]

plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')


def g(k):
    return 1/(1 + np.exp(-k))


rate = 0.002
theta = np.zeros(x.shape[1])

e = 1e-6
f = 1e-10
pre = 0
it = 0
while 1:
    now = 0
    it = it + 1
    w = np.zeros(x.shape[1])
    for i in range(tot):
        tmp = g(np.dot(theta, x[i]))
        if tmp < f:
            tmp = f
        if tmp > (1 - f):
            tmp = 1 - f
        now = now + (-y[i]*np.log(tmp) - (1 - y[i])*np.log(1 - tmp))
        z = (tmp - y[i]) * x[i]
        w += z
    w /= tot
    now /= tot
    print(np.abs(now - pre))
    if np.abs(now - pre) <= e:
        break
    pre = now
    theta -= rate*w
print(it)
print(theta)
x1 = np.linspace(0, 70, 70)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.show()

X = np.zeros(x.shape[1])
X[0] = X[0] + 20
X[1] = X[1] + 80

print(g(np.dot(theta, X)))

