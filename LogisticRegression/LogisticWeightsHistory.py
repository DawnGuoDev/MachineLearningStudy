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
记录随机梯度下降过程中的历史权重值
'''
def stocGradAscent0WeightHistory(dataList, labelList):
    dataMat = np.array(dataList)  # x数据转化为矩阵
    m, n = np.shape(dataMat)
    alpha = 0.01
    times = 100
    weights = np.ones(n)
    weightsHistory = np.zeros((times*m, n) )
    for j  in range(times):
        for i in range(m):
            h = sigmoid(np.sum(dataMat[i] * weights))
            error = (h - labelList[i])
            weights = weights - alpha * error * dataMat[i]
            weightsHistory[j*m+i, :] = weights
    return weightsHistory


'''
记录改进的随机梯度下降过程中的历史权重值
'''
def stocGradAscent1WeightHistory(dataList, labelList, numIter=150):
    dataArr = np.array(dataList)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    weightsHistory = np.zeros((numIter*m, n))
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 步长为动态变化
            rand = int(np.random.uniform(0, len(dataIndex)))
            choseIndex = dataIndex[rand]
            h = sigmoid(np.sum(dataArr[choseIndex] * weights))
            error = h - labelList[choseIndex]
            weights = weights - alpha * error * dataArr[choseIndex]
            weightsHistory[j*m+i, :] = weights
            del (dataIndex[rand])
    return weightsHistory

'''
绘制历史权值图
'''
def plotSDGError(weightsHistory):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(weightsHistory[:,0])
    plt.ylabel('X0')
    ax = fig.add_subplot(312)
    ax.plot(weightsHistory[:,1])
    plt.ylabel('X1')
    ax = fig.add_subplot(313)
    ax.plot(weightsHistory[:,2])
    plt.ylabel('X2')
    plt.xlabel(u'迭代次数')
    plt.show()


def main():
    dataList, labelList = loadDataSet("testSet.txt")

    # weightsHistory = stocGradAscent0WeightHistory(dataList, labelList)
    weightsHistory = stocGradAscent1WeightHistory(dataList, labelList, 40)
    plotSDGError(weightsHistory)


if __name__ == '__main__':
    main()
