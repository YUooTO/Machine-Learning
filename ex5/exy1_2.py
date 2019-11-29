# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def RBF_kernel(x1, x2, gama = 1000):
    return np.exp(-gama*np.dot(x1-x2, x1-x2))


class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(self.kernel)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None or self.C == 0:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])
        sv = a > 1e-5     # return a list with bool values
        ind = np.arange(len(a))[sv]  # sv's index
        self.a = a[sv]
        self.sv = X[sv]  # sv's data
        self.sv_y = y[sv]  # sv's labels
        print("%d support vectors out of %d points" % (len(self.a), n_samples))


        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl
    def loadSet(fileName):
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():  # 逐行读取，滤除空格等
            lineArr = line.strip().split(' ')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
            labelMat.append(float(lineArr[2]))  # 添加标签
        dataMat = np.array(dataMat)
        labelMat = np.array(labelMat)
        return dataMat, labelMat


    def plot_contour(X1_train, X2_train, clf):
        # 作training sample数据点的图
        pl.plot(X1_train[:,0], X1_train[:,1], "+")
        pl.plot(X2_train[:,0], X2_train[:,1], "o")
        # 做support vectors 的图
        # pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
        X1, X2 = np.meshgrid(np.linspace(-1,1,50), np.linspace(-1,1,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        # pl.contour做等值线图
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_non_linear():
        X_train, y_train = loadSet('training_3.text')
        clf = SVM(RBF_kernel)
        clf.fit(X_train, y_train)

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    test_non_linear()