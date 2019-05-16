import numpy as np
import random
import matplotlib.pyplot as plt
import math

'''
针对一个点的局部线性回归
'''
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weight = np.mat(np.eye(m))  # 对角都是1的矩阵
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weight[i, i] = np.exp((diffMat * diffMat.T) / (-2 * k ** 2))
    xTx = xMat.T * weight * xMat
    if np.linalg.det(xTx) != 0:
        theta = xTx.I * xMat.T * weight * yMat
        return theta
    return None


def lwlrAll(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        thetaRe = lwlr(testArr[i], xArr, yArr, k)
        yHat[i] = testArr[i] * thetaRe
    return yHat


def main():
    # 产生用于训练的数据
    x = np.arange(0, 25, 0.5)
    xArr = []
    yArr = []
    for xi in x:
        xArr.append([1, xi])
        yArr.append(0.5 * xi + 3 + random.uniform(0, 1) * math.sin(xi))
    xArr = np.array(xArr)
    yArr = np.array(yArr)

    # 对所有样本依次进行局部线性回归，并画图
    plt.figure(1)
    yHat = lwlrAll(xArr, xArr, yArr, 1.0)
    plt.scatter(xArr[:, 1], yArr, 60)
    plt.plot(xArr[:, 1], yHat, 'r', lw=2)

    plt.figure(2)
    yHat = lwlrAll(xArr, xArr, yArr, 0.3)
    plt.scatter(xArr[:, 1], yArr, 60)
    plt.plot(xArr[:, 1], yHat, 'r', lw=2)

    plt.figure(3)
    yHat = lwlrAll(xArr, xArr, yArr, 0.1)
    plt.scatter(xArr[:, 1], yArr, 60)
    plt.plot(xArr[:, 1], yHat, 'r', lw=2)

    plt.show()


if __name__ == "__main__":
    main()
