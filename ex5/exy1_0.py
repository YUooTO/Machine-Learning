# -*- coding: utf-8 -*-
import numpy as np
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def fit(X, y, kernel=linear_kernel, C=1e-8):
    n_samples, n_features = X.shape
    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = linear_kernel(X[i], X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1,n_samples))
    b = cvxopt.matrix(0.0)

    if C is None:
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    a1 = np.ravel(solution['x'])

    tmp = a1 > 0     # return a list with bool values
    ind = np.arange(len(a1))[tmp]  # sv's index
    a = a1[tmp]
    sv = X[tmp]  # sv's data
    sv_y = y[tmp]  # sv's labels
    print("%d support vectors out of %d points" % (len(a), n_samples))
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n],tmp])
    b /= len(a)

    # Weight vector
    if kernel == linear_kernel:
        w = np.zeros(n_features)
        for n in range(len(a)):
            # linear_kernel相当于在原空间，故计算w不用映射到feature space
            w += a[n] * sv_y[n] * sv[n]
    else:
        w = None
    return w,b,sv_y,sv


def project(X, w, b, sv_y, sv):
    if w is not None:
        return np.dot(X, w) + b
    else:
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a1, sv_y1, sv1 in zip(a, sv_y, sv):
                s += a1 * sv_y1 * linear_kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + b

def predict(X, w, b, sv_y, sv):
    return np.sign(project(X, w, b, sv_y, sv))


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


    # 仅仅在Linears使用此函数作图，即w存在时
    def plot_margin(X1_train, X2_train, w, b):
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:, 0], X1_train[:, 1], "+")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "o")
        # pl.scatter(sv[:, 0], sv[:, 1], s=100, c="g")

        xx = np.linspace(0, 200, 500)
        yy = [f(i, w, b) for i in xx]
        yy = np.array(yy)
        pl.plot(xx, yy, "k")
        pl.axis("tight")
        pl.show()


    def test_linear():
        X_train, y_train = loadSet('training_1.txt')
        X_test, y_test = loadSet('test_1.txt')
        w, b, sv_y, sv = fit(X_train, y_train)
        y_predict = predict(X_test, w, b, sv_y, sv)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct %f%%" % (correct, len(y_predict), (correct / len(y_predict))*100))
        plot_margin(X_train[y_train == 1], X_train[y_train == -1], w, b)
        plot_margin(X_test[y_test == 1], X_test[y_test == -1], w, b)
    test_linear()
