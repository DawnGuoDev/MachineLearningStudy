import numpy as np
import matplotlib.pyplot as plt

'''
从文件中读取数据
'''
def loadDataSet(filePath):
    dataList = []
    labelList = []
    f = open(filePath)
    for line in f.readlines():
        lineArr = line.strip().split()
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelList.append(int(lineArr[2]))
    return dataList, labelList


'''
sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


'''
梯度下降函数
'''
def gradAscent(dataList, labelList):
    dataMat = np.mat(dataList)  # x数据转化为矩阵
    labelMat = np.mat(labelList).T  # y数据转换为矩阵并转置为列向量
    m, n = np.shape(dataMat)  # 返回矩阵的大小
    alpha = 0.001  # 步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 权重列向量
    # 梯度下降算法
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = h - labelMat
        weights = weights - alpha * dataMat.T * error
    return weights


'''
随机梯度下降算法
'''
def stocGradAscent0(dataList, labelList):
    dataArr = np.array(dataList)  # x数据转化为矩阵
    m, n = np.shape(dataArr)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataArr[i]*weights))
        error = (h-labelList[i])
        weights = weights-alpha*error*dataArr[i]
    return weights

'''
改进的随机梯度下降算法
'''
def stocGradAscent1(dataList, labelList, numIter=150):
    dataArr = np.array(dataList)
    m,n = np.shape(dataArr)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01        # 步长为动态变化
            rand = int(np.random.uniform(0, len(dataIndex)))
            choseIndex = dataIndex[rand]
            h = sigmoid(np.sum(dataArr[choseIndex]*weights))
            error = h-labelList[choseIndex]
            weights = weights-alpha*error*dataArr[choseIndex]
            del(dataIndex[rand])
    return weights


'''
画分类的分割线
'''
def plotBestFit(dataList, labelList, weights):
    dataArr = np.array(dataList)
    m = np.shape(dataArr)[0]  # 获取样本的数量
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    # 对不同标记的数据进行分类
    for i in range(m):
        if int(labelList[i]) == 0:
            xcord0.append(dataArr[i, 1])
            ycord0.append(dataArr[i, 2])
        else:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
    # 画散点图
    plt.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    plt.scatter(xcord1, ycord1, s=30, c='green')
    # 画分割线
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = (-weights[0] - weights[1] * x1) / weights[2]
    plt.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def main():
    dataList, labelList = loadDataSet("testSet.txt")
    # weights = gradAscent(dataList, labelList)
    # weights = stocGradAscent0(dataList, labelList)
    weights = stocGradAscent1(dataList, labelList)
    plotBestFit(dataList, labelList, weights)



if __name__ == '__main__':
    main()
