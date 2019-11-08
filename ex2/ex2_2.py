import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("ex2x.dat")
y = np.loadtxt("ex2y.dat").reshape((-1, 1))


def g(k):
    return 1/(1 + np.exp(-k))


x = np.hstack((np.ones([80, 1]), x))
x_pos = x[:40, :]
x_neg = x[40:, :]
plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')

it = 261314
theta = [-11.0487, 0.109761, 0.103191]
x1 = np.linspace(10, 70, 70)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.title("0.002")
plt.show()
print(1 - g(np.dot([1, 20, 80], theta)))


rate = 0.001
theta = [-2.53826, 0.0584452, 0.00686168]
it = 90758
plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')

x1 = np.linspace(10, 70, 70)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.title("0.001")
plt.show()

rate = 0.0018
theta = [-8.70063, 0.0939862, 0.0776859]
it = 205439
plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')

x1 = np.linspace(10, 70, 70)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.title("0.0018")
plt.show()

rate = 0.0022
theta = [-13.1825, 0.124879, 0.125935]
it = 322308
plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')

x1 = np.linspace(10, 70, 70)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.title("0.0022")
plt.show()

rate = 0.0025
theta = [-16.1544, 0.146911, 0.157049]
it = 559615
plt.scatter([b[1] for b in x_pos], [b[2] for b in x_pos], marker='+')
plt.scatter([b[1] for b in x_neg], [b[2] for b in x_neg], marker='o')
x1 = np.linspace(10, 70, 60)
y = [xx*(-1*theta[1]/theta[2]) - theta[0]/theta[2] for xx in x1]
plt.plot(x1, y)
plt.title("0.0025")
plt.show()


