import numpy as np
import re


'''
将读取的内容进行解析，返回文档解析后的词条列表
'''
def textParse(bigString):
    listOfTokens = re.split("\W*", bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


'''
读取文本内容    
'''
def loadDateSet():
    docList = []    # 所有文档解析后的词条集合
    classList = []  # 文档对应的类别，spam为1，ham为0

    for i in range(1, 26):
        with open("email/spam/%d.txt"%i) as f:
            wordsList = textParse(f.read())
            docList.append(wordsList)
            classList.append(1)
        with open("email/ham/%d.txt"%i) as f:
            wordsList = textParse(f.read())
            docList.append(wordsList)
            classList.append(0)
    return docList, classList


'''
训练集和测试集分开
'''
def testTrainSplit(docList):
    trainListIndex = list(range(len(docList)))
    testListIndex= []

    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainListIndex)))
        testListIndex.append(trainListIndex[randIndex])
        del(trainListIndex[randIndex])
    return trainListIndex, testListIndex


'''
创建文档词汇表
'''
def createVocabList(dataList):
    vocabSet = set([])
    for words in dataList:
        vocabSet = vocabSet | set(words)
    return list(vocabSet)


'''
创建词向量
'''
def wordsToVec(vocabList, inputDoc):
    wordVec = [0] * len(vocabList)
    for word in inputDoc:
        if word in vocabList:
            wordVec[vocabList.index(word)] += 1  # += 1
    return wordVec


'''
朴素贝叶斯的训练
'''
def trainNB(trainMat, trainClass):
    numTrain =  len(trainMat)
    numWords = len(trainMat[0])
    pSpam = np.sum(trainClass)/float(numTrain)
    numSpam = np.ones(numWords)
    numHam = np.ones(numWords)
    spamDenom = 2.0
    hamDenom = 2.0
    for i in range(numTrain):
        if trainClass[i] == 1:
            numSpam += trainMat[i]
            spamDenom += np.sum(trainMat[i])
        else:
            numHam += trainMat[i]
            hamDenom += np.sum(trainMat[i])

    spam1Vec = np.log(numSpam/spamDenom)
    ham1Vec = np.log(numHam/hamDenom)
    return spam1Vec, ham1Vec, pSpam

'''
朴素贝叶斯进行决策
'''
def classifyNB(inputDoc, spam1Vec, ham1Vec, pSpamTrain):
    pSpam = np.sum(inputDoc * spam1Vec) + np.log(pSpamTrain)
    pHam = np.sum(inputDoc * ham1Vec) + np.log(1-pSpamTrain)
    if pSpam > pHam:
        return 1
    else:
        return 0


def main():
    docList, classList = loadDateSet()  # 文档总的词条列表，类别列表
    vocabList = createVocabList(docList)  # 文档的词汇集

    sumErrorRate = 0.0  # 总的错误率
    for k in range(10):
        trainListIndex, testListIndex = testTrainSplit(docList)  # 训练集、测试集的index
        # 获取训练集词条向量
        trainWordVecs = []  # 训练集词向量
        trainClass = []  # 训练集词向量类别
        for i in trainListIndex:
            trainWordVecs.append(wordsToVec(vocabList, docList[i]))
            trainClass.append(classList[i])

        # 训练，获取所需概率值
        spam1Vec, ham1Vec, pSpamTrain = trainNB(np.array(trainWordVecs),
                                                np.array(trainClass))

        errorCount = 0  # 测试集上的错误次数
        for docIndex in testListIndex:
            wordVec = wordsToVec(vocabList, docList[docIndex])
            if classList[docIndex] != int(classifyNB(
                    np.array(wordVec), spam1Vec, ham1Vec, pSpamTrain)):
                errorCount += 1
                print("第%d次实验中，错分的词列表为：%s"%(k, docList[docIndex]))
        sumErrorRate += errorCount / len(testListIndex)
        print("第%d次实验中，在测试集上的错误率为%f" % (k, errorCount / len(testListIndex)))
    print("在10次实验过程中，测试集上的平均错误率为%f"%(sumErrorRate/float(10)))


if __name__ == '__main__':
    main()

