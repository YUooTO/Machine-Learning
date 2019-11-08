import matplotlib.pyplot as plt
import numpy as np
from numpy import *

fp = open("ex2x.dat", "r");
fq = open("ex2y.dat", "r");

x = []
y = []
for i in fp.readlines():
	x.append([1, float(i.split()[0]), float(i.split()[1])])


for i in fq.readlines():
	y.append(float(i.strip()))


plt.scatter([i[1] for i in x[:40]], [i[2] for i in x[:40]], c = 'g')
plt.scatter([i[1] for i in x[40:]], [i[2] for i in x[40:]], c = 'r')

x = (x - np.mean(x)) / np.std(x)


def g(z):
	return 1 / (1 + exp(-z))

theta = np.array([0.0, 0.0, 0.0])
X = np.array(x)
Y = np.array(y)
m = len(x)
iteras = 20
J = []
for it in range(iteras):
	# print(theta)
	rt = g(X.dot(theta))
	J_ = 1 / m * X.T.dot(rt - Y)
	H = 1 / m * X.T.dot(np.diag(rt)).dot(np.diag(1 - rt)).dot(X)
	j = 1 / m * np.sum(- Y * log(rt) - (1 - Y) * log(1 - rt))
	J.append([it, j])
	Z = mat(H).I.dot(J_)
	z = Z.tolist()
	for i in range(3):
		theta[i] -= z[0][i]
	#print(theta.shape)

plt.plot([0, - theta[0] / theta[1]], [- theta[0] / theta[2], 0], 'b')
plt.show()

plt.plot([x[0] for x in J], [x[1] for x in J])
#plt.show()

print("theta", theta)
print(g(np.array([1, 20, 80]).dot(theta)))



