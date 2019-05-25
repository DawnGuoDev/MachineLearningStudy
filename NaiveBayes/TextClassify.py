import numpy as np


'''
:return
    postingList：文档被切分成一系列词条集合后的文档集合
    classVec：类别标签的集合，分为侮辱性和非侮辱性两类
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字, 0 not
    return postingList, classVec


'''
创建文档集合的词汇表
:return 
    list(vocalSet):词汇表
'''
def createVocabList(docSet):
    vocabSet = set([])      # 词汇表Set集合
    for doc in docSet:      # 遍历得到词汇表Set集合
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


'''
把词条集合转换成词条向量
:return
    wordsVec:词向量
'''
def wordsToVec(vocabList, inputWords):
    wordsVec = [0] * len(vocabList)
    for word in inputWords:
        if word in vocabList:
            wordsVec[vocabList.index(word)] = 1 # +=1 变成词袋模型
    return wordsVec


'''
朴素贝叶斯的训练，即获取所需的条件概率;
暂时只支持二分类
'''
def trainNB(dataTrain, classTrain):
    numTrainDoc = len(dataTrain)                    # 训练集的数量
    numWords = len(dataTrain[0])                    # 一条词向量的数量
    p1Train = np.sum(classTrain)/float(numTrainDoc)      # p1的概率
    p0Num = np.ones(numWords)                       # 存放0类中各词汇数的array
    p1Num = np.ones(numWords)                       # 存放1类中各词汇数的array
    p0Denom = 2.0                                   # 分母，0类中总词汇数
    p1Denom = 2.0                                   # 分母，1类中总词汇数
    for i in range(numTrainDoc):
        if classTrain[i] == 0:
            p0Num += dataTrain[i]
            p0Denom += np.sum(dataTrain[i])
        else:
            p1Num += dataTrain[i]
            p1Denom += np.sum(dataTrain[i])
    p0Vec = np.log(p0Num/p0Denom)
    p1Vec = np.log(p1Num/p1Denom)
    return p0Vec, p1Vec, p1Train

'''
使用朴素贝叶斯进行决策
:input
    wordsVec:要分类的向量
'''
def classifyNB(wordsVec, p0Vec, p1Vec, p1Train):
    p0 = np.sum(wordsVec*p0Vec) + np.log(1-p1Train) # 获取属于0类的概率
    p1 = np.sum(wordsVec*p1Vec) + np.log(p1Train)   # 获取属于1类的概率
    if p0 < p1:
        return 1
    else:
        return 0


def main():
    postList, postClasses = loadDataSet()   # 获取词条列表和类别向量
    vocalList = createVocabList(postList)   # 获取词汇表
    # 获取词汇向量
    trainWordsVecs = []
    for post in postList:
        trainWordsVecs.append(wordsToVec(vocalList, post))
    # 训练朴素贝叶斯
    p0Vec, p1Vec, p1Train = trainNB(np.array(trainWordsVecs), np.array(postClasses))
    # 测试
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = wordsToVec(vocalList, testEntry)
    print('%s 被分类为:%d'%(testEntry, classifyNB(np.array(thisDoc), p0Vec, p1Vec, p1Train)))
    testEntry = ['stupid', 'garbage']
    thisDoc = wordsToVec(vocalList, testEntry)
    print('%s 被分类为:%d' % (testEntry, classifyNB(np.array(thisDoc), p0Vec, p1Vec, p1Train)))

if __name__ == "__main__":
    main()

