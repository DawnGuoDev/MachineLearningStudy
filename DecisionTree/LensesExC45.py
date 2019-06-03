import operator
import math
import TreePlot


'''
从文件中读取数据集
'''
def loadDataSet(filePath):
    fr = open(filePath)
    lensesLable = ['age', 'prescript', 'astigmatic', 'tearRate']    # 属性名
    lensesData = [] # 隐形眼镜数据
    # 读取文件
    for line in fr.readlines():
        data = line.strip().split("\t")
        lensesData.append(data)
    return lensesData, lensesLable


'''
选取数量最多的类别
'''
def majorityClass(classList):
    classCount = {} # 类别计数 类别:数量
    # 遍历进行计数
    for value in classList:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1

    # 选择数量最多的类别
    classSort = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return classSort[0][0]


'''
数据集进行划分
'''
def splitDataSet(dataSet, index, value):
    subDataSet = []
    for data in dataSet:
        if data[index] == value:
            subData = data[:index]
            subData.extend(data[index+1:])
            subDataSet.append(subData)
    return subDataSet


'''
计算信息嫡
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 数据集长度
    classList = [data[-1] for data in dataSet]  # 数据集的类别集合
    classCount = {} # 类别计数
    shannonEnt = 0.0   # 信息嫡
    # 遍历计数
    for value in classList:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1

    # 计算信息嫡
    for key in classCount.keys():
        prob = float(classCount[key])/numEntries
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt



'''
选择最好的属性，C4.5
'''
def chooseBestFeature(dataSet):
    numEntries = len(dataSet)   # 数据集数量
    numFeatures = len(dataSet[0]) - 1 # 属性的数量
    baseEnt = calcShannonEnt(dataSet)   # 整个数据集的香农嫡
    bestFeature = -1    # 选出的最好属性的index
    infoGains = []  # 存储各属性的信息增益
    gainRatios = []  # 存储各属性的增益率
    maxGainRatio = 0.0 # 增益率最大
    for i in range(numFeatures):
        newEnt = 0.0    # 划分之后的信息嫡之和
        numIV = 0.0     # 属性的固有值
        featList = [data[i] for data in dataSet]    # 获取所有该属性的所有值
        featSet = set(featList) # 获取该属性不同的值
        for value in featSet:
            subDataSet = splitDataSet(dataSet, i, value)    # 获取属性值相同的数据集
            prob = float(len(subDataSet)) / numEntries
            numIV -= prob * math.log(prob, 2)
            newEnt += prob * calcShannonEnt(subDataSet)
        newGain = baseEnt - newEnt  # 划分之后的信息增益
        newGainRatio = newGain/numIV    # 划分之后的增益率
        infoGains.append(newGain)
        gainRatios.append(newGainRatio)

    infoGainAvg = sum(infoGains) / len(infoGains) # 平均信息增益率

    for i in range(len(infoGains)):
        if (infoGains[i] >= infoGainAvg) and (gainRatios[i] > maxGainRatio):
            maxGainRatio = gainRatios[i]
            bestFeature = i

    return bestFeature


'''
判断样本在属性集合上取值是否相同
'''
def propertyIsSame(dataSet):
    numFeatures = len(dataSet[0])-1
    for i in range(numFeatures):
        temp = dataSet[0][i]
        for data in dataSet:
            if data[i] != temp:
                return 0
    return 1


'''
递归创建决策树
'''
def createTree(dataSet, labels):
    classList = [data[-1] for data in dataSet]  # 获取数据集中所属类别的数据
    # 检测数据集是否符合同一个分类，相同则返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 假如属性用完了，那么选择当前数据集中数量最多的类别
    if (len(dataSet[0]) == 1)  or propertyIsSame(dataSet):
        return majorityClass(classList)

    # 选取最好的属性，采取C4.5
    bestFeature = chooseBestFeature(dataSet)    # 最好属性的index
    bestFeatLabel = labels[bestFeature] # 最好属性的名字
    decisionTree = {bestFeatLabel:{}}   # 存储决策树
    featValues = [data[bestFeature] for data in dataSet]
    featValuesSet = set(featValues)   # 不同类别的集合
    del(labels[bestFeature])
    for value in featValuesSet:
        subLabels = labels[:]   # 针对划分之后的数据集都要有一个新的labels
        decisionTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature,
                                                                     value), subLabels)
    return decisionTree


'''
使用决策树进行预测
'''
def classifyDecisionTree(tree, labels, testData):
    firstLabel = list(tree.keys())[0]   # 获取树的第一决策节点的属性
    firstLabelIndex = labels.index(firstLabel)  # 属性对应的index
    secondDict = tree[firstLabel]
    value = testData[firstLabelIndex]   #  属性值

    if type(secondDict[value]).__name__ == "dict":  # 假如还是一棵树则递归
        classLabel = classifyDecisionTree(secondDict[value], labels, testData)
    else:
        classLabel = secondDict[value]

    return classLabel


if __name__ == '__main__':
    # 读取数据
    lensesData, lensesLable = loadDataSet("lenses.txt")

    # 复制属性标签，
    # createTree()操作会影响传入的类别标签
    lensesLableCopy = lensesLable[:]

    # 创建树
    decisionTree = createTree(lensesData, lensesLableCopy)
    print(decisionTree)

    # 对树进行绘图
    TreePlot.createPlot(decisionTree)

    # 进行预测
    classLabel = classifyDecisionTree(decisionTree, lensesLable,
                                      ["young", "hyper", "yes", "normal"])
    print(classLabel)