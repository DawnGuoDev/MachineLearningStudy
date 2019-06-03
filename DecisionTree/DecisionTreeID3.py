import math
import operator
import pickle
import TreePlot


'''
创建数据集
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['生存', '脚蹼']
    return dataSet, labels


'''
信息嫡的计算
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 数据集的数量
    labelCounts = {}            # label及其数量
    # 统计label及对应的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 计算香农嫡
    shannonEnt = 0.0    # 香农嫡
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


'''
划分出分类后的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


'''
获取数量最多的类别
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


'''
选择最好的特征
'''
def chooseBestFeature(dataSet):
    numEntries = len(dataSet)
    numFeatures = len(dataSet[0])-1 # 特征数
    baseEnt = calcShannonEnt(dataSet)   # 总的信息嫡
    baseInfoGain = 0.0          # 信息增益
    bestFeature = -1            # 最好的特征的Index
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取该特征的所有特征值
        uniqueVals = set(featList)     # 获取特征的不同特征值集合
        newEnt = 0.0 # 按照此特征进行分类的信息嫡之和
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet)/float(numEntries)
            newEnt += prob * calcShannonEnt(subDataSet)
        infoGain = baseEnt - newEnt
        if infoGain > baseInfoGain :
            baseInfoGain = infoGain
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
递归创建树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    # 获取类别列表

    # 假如都是同一类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 所有属性都已经遍历完或者样本在属性上相同
    if (len(dataSet[0]) == 1) or propertyIsSame(dataSet):
        return majorityCnt(classList)

    bestFeat = chooseBestFeature(dataSet)
    bestFeatLable = labels[bestFeat]
    decisionTree = {bestFeatLable:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    del (labels[bestFeat])
    for value in uniqueVals:
        subLabels = labels[:]
        decisionTree[bestFeatLable][value]  = createTree(splitDataSet(dataSet, bestFeat, value),
                                                         subLabels)
    return decisionTree


'''
测试决策树
'''
def classifyDecisionTree(tree, featLabels, testVec):
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    featIndex= featLabels.index(firstStr)
    if type(secondDict[testVec[featIndex]]).__name__ == 'dict':
        classLabel = classifyDecisionTree(secondDict[testVec[featIndex]], featLabels, testVec)
    else:
        classLabel = secondDict[testVec[featIndex]]
    return classLabel


'''
存储序列化对象
'''
def storeTree(tree, filename):
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()


'''
读取序列化对象
'''
def grabTree(filename):
    fr = open(filename, "rb")
    return pickle.load(fr)


if __name__ == '__main__':

    # 创建示例数据集
    dataSet, labels = createDataSet()
    lebelsCopy = labels[:]

    # 学习构建决策树
    tree = createTree(dataSet, labels)
    print(tree)

    # 画决策树
    TreePlot.createPlot(tree)

    # 序列化存储树结构
    storeTree(tree, "object.txt")
    # 文件中读取数结构
    myTree = grabTree("object.txt")
    print(myTree)
    print(classifyDecisionTree(myTree, lebelsCopy, [1, 1]))
