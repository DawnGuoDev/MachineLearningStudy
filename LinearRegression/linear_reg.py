import numpy as np
import matplotlib.pyplot as plt
import random
import math

'''
批量梯度下降算法
'''
def BGD(xArr, yArr):
    theta = np.array([0.0, 0.0])  # 初始值
    alpha = 0.005  # 步长
    times = 10000
    m = len(xArr)

    for i in range(times):
        # theta的一次梯度下降
        for j in range(len(theta)):
            sum0 = 0
            # 求和
            for xk, yk in zip(xArr, yArr):
                sum0 += ((theta * xk).sum() - yk) * xk[j]
            # 梯度下降
            theta[j] = theta[j] - (alpha * sum0) / m
            # 不除以m的话，得把步长设置的很小

    return theta


'''
随机梯度下降算法
'''
def SGD(xArr, yArr):
    theta = np.array([6.0, 6.0])  # 初始值
    alpha = 0.005  # 步长
    times = 20000
    m = len(xArr)
    for i in range(times):
        for j in range(len(theta)):
            # sum0 = 0
            # 求和
            for xk, yk in zip(xArr, yArr):
                # 一次梯度下降
                sum0 = ((theta * xk).sum() - yk) * xk[j]
                theta[j] = theta[j] - (alpha * sum0) / m

    return theta


'''
小批量梯度下降算法
'''
def MBGD(xArr, yArr):
    theta = np.array([6.0, 6.0])  # 初始值
    alpha = 0.005  # 步长
    times = 20000
    m = len(xArr)
    n = 5

    for i in range(times):
        train_list = []
        for t in range(n):
            train_list.append(random.randint(0, m - 1))
        for j in range(len(theta)):
            sum0 = 0
            # 一次梯度下降
            for k in train_list:
                sum0 += ((theta * xArr[k]).sum() - yArr[k]) * xArr[k][j]
            theta[j] = theta[j] - (alpha * sum0) / m

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


def main():
    x = np.arange(0, 25, 0.5)
    xArr = []
    yArr = []
    for xi in x:
        xArr.append([1, xi])  # x向量
        yArr.append(0.5 * xi + 3 + random.uniform(0, 1) * math.sin(xi))  # y向量的值
    xArr = np.array(xArr)
    yArr = np.array(yArr)

    theta1 = BGD(xArr, yArr)
    print("批量梯度下降算法下求的theta值", theta1)

    # print("未归一化下x=0.5的预测值",(theta1*np.array([1, 0.5])).sum())
    # '''归一化测试'''
    # ymax, ymin = yArr.max(), yArr.min()
    # xmax, xmin = xArr[:, 1].max(), xArr[:, 1].min()
    # xArr = (xArr - xmin) / (xmax - xmin)
    # yArr = (yArr - ymin) / (xmax - xmin)
    #
    # theta1 = BGD(xArr, yArr)
    # print("归一化下批量梯度下降算法下求的theta值", theta1)
    # print("归一化下x=0.5的预测值", ((theta1 * np.array([1, (0.5-xmin)/(xmax-xmin)])).sum())*(ymax-ymin)+ymin)

    # theta2 = SGD(xArr, yArr)
    # print("随机梯度下降算法下求的theta值", theta2)
    # theta3 = MBGD(xArr, yArr)
    # print("小批量梯度下降算法下求的theta值", theta3)
    theta4 = mat(xArr, yArr)
    print("矩阵方法求的theta值", theta4)

    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for i in xArr:
        y1.append((i * theta1).sum())
        # y2.append((i * theta2).sum())
        # y3.append((i * theta3).sum())
        # y4.append((i * theta4).sum())

    # corrcoef计算预测值和真实值的相关性
    print(np.corrcoef(y1, yArr))

    # 画图显示
    plt.scatter(x, yArr, 60)
    plt.plot(x, y1, 'r', lw=2)
    # plt.plot(x, y2, 'k', lw=2)
    # plt.plot(x, y3, 'w', lw=2)
    # plt.plot(x, y4, 'y', lw=2)
    plt.show()


if __name__ == "__main__":
    main()
