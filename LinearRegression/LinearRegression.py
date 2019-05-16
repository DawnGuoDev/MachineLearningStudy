import numpy as np
import math
import random
import matplotlib.pyplot as plt


'''
批量梯度下降算法
'''
def BGD(xArr, yArr, alpha, theta, times):
    m = len(xArr)

    for k in range(times):
        # 一次迭代
        for j in range(len(theta)):
            sum = 0.0
            # 进行求和
            for xi, yi in zip(xArr, yArr):
                sum += ((theta * xi).sum() - yi) * xi[j]
            # 更新
            theta[j] = theta[j] - (alpha * sum) / m  # 不除以m的话，得把步长设置的很小很小

    return theta


'''
随机梯度下降算法
'''
def SGD(xArr, yArr, alpha, theta, times):
    m = len(xArr)

    for k in range(times * 10):
        t = random.randint(0, m - 1)
        # 一次迭代
        for j in range(len(theta)):
            sum = ((theta * xArr[t]).sum() - yArr[t]) * xArr[t][j]
            # 更新
            theta[j] = theta[j] - (alpha * sum) / m
    return theta


'''
小批量梯度下降算法
'''
def MBGD(xArr, yArr, alpha, theta, times):
    m = len(xArr)
    n = 5  # 小批量数据的数量

    for k in range(times):
        # 随机产生小批量的数据
        trainSet = []
        for t in range(n):
            trainSet.append(random.randint(0, m - 1))
        # 一次迭代
        for j in range(len(theta)):
            sum = 0.0
            for i in trainSet:
                sum += ((theta * xArr[i]).sum() - yArr[i]) * xArr[i][j]
            # 更新
            theta[j] = theta[j] - (alpha * sum) / m

    return theta


'''
矩阵的方法求参数值
'''
def mat(xArr, yArr):
    xMat = np.mat(xArr)  # x矩阵
    xMatT = xMat.T  # x的转置矩阵
    yMat = np.mat(yArr).T  # y矩阵
    xTx = xMatT * xMat
    if (np.linalg.det(xTx) != 0):
        theta = xTx.I * xMatT * yMat
        return theta
    return None


'''
画图
'''
def draw_line(xArr, theta):
    y = []
    for i in xArr:
        y.append((theta * i).sum())
    plt.plot(xArr[:, 1], np.array(y), 'r', lw=2)


def main():
    x = np.arange(0, 25, 0.5)
    xArr = []
    yArr = []
    # 产生用于训练的数据
    for xi in x:
        xArr.append([1, xi])
        yArr.append(0.5 * xi + 3 + random.uniform(0, 1) * math.sin(xi))
    xArr = np.array(xArr)
    yArr = np.array(yArr)

    alpha = 0.005  # 步长
    theta = np.array([0.0, 0.0])  # 回归系数
    times = 10000  # 迭代次数
    theta_re = BGD(xArr, yArr, alpha, theta, times)
    print("批量梯度下降算法下求的theta值 ", theta_re)
    print("批量梯度下降算法求得相关系数为\n",  # 计算相关系数
          np.corrcoef(yArr, np.array(np.mat(xArr) * np.mat(theta_re).T)[:,0]))
    draw_line(xArr, theta_re)

    theta_re = SGD(xArr, yArr, alpha, theta, times)
    print("随机梯度下降算法下求的theta值", theta_re)
    print("随机梯度下降算法求得相关系数为\n",  # 计算相关系数
          np.corrcoef(yArr, np.array(np.mat(xArr) * np.mat(theta_re).T)[:, 0]))
    draw_line(xArr, theta_re)

    theta_re = MBGD(xArr, yArr, alpha, theta, times)
    print("小批量梯度下降算法下求的theta值", theta_re)
    print("小批量梯度下降算法求得相关系数为\n",
          np.corrcoef(yArr, np.array(np.mat(xArr) * np.mat(theta_re).T)[:, 0]))
    draw_line(xArr, theta_re)

    theta_re = mat(xArr, yArr)
    print("矩阵求逆法求得的theta值", np.array(theta_re)[:, 0])
    print("小批量梯度下降算法求得相关系数为\n",
          np.corrcoef(yArr, np.array(np.mat(xArr) * theta_re)[:, 0]))
    draw_line(xArr, np.array(theta_re)[:, 0])

    plt.scatter(x, yArr, 60)
    plt.show()


if __name__ == "__main__":
    main()
