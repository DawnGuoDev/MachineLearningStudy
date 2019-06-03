import matplotlib.pyplot as plt


# 用来正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrowArgs = dict(arrowstyle="<-")
# 画图的区域
createPlotArea = None


'''
节点的绘制
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #在全局变量createPlot中绘图
    createPlotArea.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords="axes fraction",
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrowArgs)


'''
创建画图的区域以及获取区域信息
'''
def createPlot(tree):
    global createPlotArea
    #创建一个新图形
    fig = plt.figure(1, facecolor='white')
    #清空绘图区
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    #给全局变量createPlot赋值
    createPlotArea = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '')
    #显示最终绘制结果
    plt.show()


'''
递归得到叶节点的数量
'''
def getNumLeafs(tree):
    numLeafs = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1

    return numLeafs

'''
递归得到树的深度
'''
def getTreeDepth(tree):
    maxDepth = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth :
            maxDepth = thisDepth

    return maxDepth


'''
在分类的线上填充文本信息
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]    # x的坐标
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]    # y的坐标
    createPlotArea.text(xMid, yMid, txtString)

'''
递归画出数的结构
'''
def plotTree(tree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)
    firstStr = list(tree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 +  float(numLeafs))/2.0/plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


'''
测试画图效果
'''
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers':
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


if __name__ == '__main__':
    tree = retrieveTree(1)
    print(createPlot(tree))
