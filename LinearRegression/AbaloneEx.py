import numpy as np


def loadDataSet(filePath):
    f = open(filePath)
    numFeat = len(f.readline().split("\t")) - 1  # 获取一行特征的数量
    dataList = []
    labelList = []
    f.seek(0)  # 文件指针回到初始位置
    for line in f.readlines():
        lineArr = []
        curLine = line.strip().split("\t")  # strip去掉\n
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataList.append(lineArr)
        labelList.append(float(curLine[-1]))
    return dataList, labelList


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weight = np.mat(np.eye(m))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weight[i, i] = np.exp((diffMat * diffMat.T) / (-2.0 * k ** 2))

    xTx = xMat.T * (weight * xMat)

    if np.linalg.det(xTx) != 0.0:
        theta = xTx.I * (xMat.T * (weight * yMat))
        return testPoint * theta
    return None


def lwlrAll(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) != 0.0:
        theta = xTx.I * (xMat.T * yMat)
        return theta
    return None


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


if __name__ == '__main__':
    abaloneX, abaloneY = loadDataSet("abalone.txt")

    yHat01 = lwlrAll(abaloneX[0:99], abaloneX[0:99], abaloneY[0:99], 0.1)
    yHat1 = lwlrAll(abaloneX[0:99], abaloneX[0:99], abaloneY[0:99], 1)
    yHat10 = lwlrAll(abaloneX[0:99], abaloneX[0:99], abaloneY[0:99], 10)
    print("k=0.1时，训练集上的误差为：", rssError(abaloneY[0:99], yHat01))
    print("k=1时，训练集上的误差为：", rssError(abaloneY[0:99], yHat1))
    print("k=10时，训练集上的误差为：", rssError(abaloneY[0:99], yHat10))

    yHat01 = lwlrAll(abaloneX[100:199], abaloneX[0:99], abaloneY[0:99], 0.1)
    yHat1 = lwlrAll(abaloneX[100:199], abaloneX[0:99], abaloneY[0:99], 1)
    yHat10 = lwlrAll(abaloneX[100:199], abaloneX[0:99], abaloneY[0:99], 10)
    print("k=0.1时，测试集上的误差为：", rssError(abaloneY[100:199], yHat01))
    print("k=1时，测试集上的误差为：", rssError(abaloneY[100:199], yHat1))
    print("k=10时，测试集上的误差为：", rssError(abaloneY[100:199], yHat10))

    theta = standRegres(abaloneX[0:99],abaloneY[0:99])
    yHat = np.mat(abaloneX[100:199])*theta
    print("简单线性回归时的误差为：", rssError(abaloneY[100:199], yHat.T.A))
