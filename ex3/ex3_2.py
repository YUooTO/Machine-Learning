import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex3Logx.dat", delimiter=',')
y = np.loadtxt("ex3Logy.dat").reshape((-1, 1))
tot = x.shape[0]
x_pos = []
x_neg = []
for i in range(tot):
    if y[i]:
        x_pos.append(x[i])
    else:
        x_neg.append(x[i])
x_pos = np.asarray(x_pos)
x_neg = np.asarray(x_neg)

plt.scatter([b[0] for b in x_pos], [b[1] for b in x_pos], marker='+')
plt.scatter([b[0] for b in x_neg], [b[1] for b in x_neg], marker='o')
# plt.show()

X = []
for it in x:
    z = []
    for j in range(7):
        for k in range(j+1):
            z.append((it[0] ** k) * (it[1] ** (j - k)))
    X.append(z)
X = np.asarray(X)
Z = np.zeros([28, 28])
Z = np.mat(Z)

for i in range(1, 28):
    Z[i, i] = 1
Y = y


def g(x):
    return 1 / (1 + np.exp(-x))


rate = 10
theta = np.zeros([28, 1])
theta = np.mat(theta)
X = np.mat(X)
Y = np.mat(Y)

iterator = 20
for it in range(iterator):
    now = 0
    J_ = np.mat(np.zeros([28, 1]))
    now = np.mat(np.zeros([28, 28]))
    J = 0
    for i in range(tot):
        tmp = g(np.dot(X[i], theta)[0, 0])
        J += (Y[i, 0] * np.log(tmp) + (1 - Y[i, 0]) * np.log(1 - tmp))
        now = now + (tmp * (1 - tmp) * (X[i].T * X[i]))
        for j in range(28):
            J_[j, 0] += (tmp - Y[i, 0]) * X[i, j]

    J_ /= tot
    now /= tot
    J /= -tot
    H = now + (rate / tot) * Z
    for i in range(1, 28):
        J_[i][0] += (rate / tot) * theta[i]
        J += rate/(2 * tot) * theta[i]**2
    print(J)
    Z0 = H.I * J_
    theta -= Z0
print(theta.T)
xx = np.linspace(-1, 1.5, 200)
yy = np.linspace(-1, 1.5, 200)

zz0 = []
for i in xx:
    zz = []
    for j in yy:
        z = []
        for l in range(7):
            for k in range(l + 1):
                z.append((i ** k) * (j ** (l - k)))
        zz.append(np.array(z).dot(theta))
    zz0.append(zz)

X, Y = np.meshgrid(xx, yy)
Z = np.array(zz0)
Z = np.mat(Z)
Z = Z.T
plt.contour(X, Y, Z, [0])
plt.show()
