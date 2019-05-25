import re
import numpy as np


'''
创建包含所有文档中出现的单词不重复的列表
'''
def createVocabList(dataList):
    vocabSet = set([])
    for document in dataList:
        vocabSet = vocabSet | set(document) # 做并集操作
    return list(vocabSet)


'''
邮件内容处理成列表
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


'''
转化为文档向量
'''
def setOfWordsToVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # +=1 变成词袋模型
    return returnVec


'''
训练朴素贝叶斯
'''
def trainNB(trainMat, trainClasses):
    numTrainDoc = len(trainMat)
    numWords = len(trainMat[0])
    pSpam = np.sum(trainClasses)/float(numTrainDoc)
    spamNum = np.ones(numWords)
    hamNum = np.ones(numWords)
    spamDenom = np.sum(trainClasses) + 2
    hamDenom = len(trainClasses) - np.sum(trainClasses) + 2
    for i in range(numTrainDoc):
        if trainClasses[i] == 1:
            spamNum += trainMat[i]
        else:
            hamNum += trainMat[i]
    spam1Vec = np.log(spamNum / spamDenom)
    ham1Vec = np.log(hamNum / hamDenom)
    return spam1Vec, ham1Vec, pSpam


'''
朴素贝叶斯进行决策
'''
def classifyNB(vecToClassify, spam1Vec, ham1Vec, pSpamTrain):
    pSpam = 0.0
    pHam = 0.0
    for i in range(len(vecToClassify)):
        if vecToClassify[i] == 1:
            pSpam +=  spam1Vec[i]
            pHam +=  ham1Vec[i]
        else:
            pSpam +=  (1 - spam1Vec[i])
            pHam +=  (1 - ham1Vec[i])
    pSpam +=  np.log(pSpamTrain)
    pHam += np.log(1 - pSpamTrain)
    if pSpam > pHam:
        return 1
    else:
        return 0

def main():
    docList = []
    classList = []
    fullText = []
    # 读取文件内容
    for i in range(1, 26):
        with open('email/spam/%d.txt'%i) as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            # 注意这是extend,把wordList中的元素添加到fullText中
            fullText.extend(wordList)
            classList.append(1)
        with open('email/ham/%d.txt'%i) as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)

    sumErrorRate = 0.0
    for k in range(10):
        # 分成训练集和测试集
        trainingList = list(range(50))          # 训练集的下标
        testList = []                           # 测试集的下标
        for i in range(10):
            randIndex = int(np.random.uniform(0, len(trainingList)))
            testList.append(trainingList[randIndex])
            del(trainingList[randIndex])

        # 进行训练
        trainingMat = []
        trainingClasses = []
        vocabList = createVocabList(docList)  # 获取没重复的词条列表
        for docInex in trainingList:
            trainingMat.append(setOfWordsToVec(vocabList, docList[docInex]))
            trainingClasses.append(classList[docInex])
        spam1Vec, ham1Vec, pSpam = trainNB(np.array(trainingMat), np.array(trainingClasses))


        # 测试集上进行测试
        errorCount = 0
        for docInex in testList:
            wordVector = setOfWordsToVec(vocabList, docList[docInex])
            if classifyNB(np.array(wordVector), spam1Vec, ham1Vec, pSpam) != classList[docInex]:
                errorCount += 1
                print(docList[docInex])
        print("错误率是%f"%(float(errorCount)/10))
        sumErrorRate += float(errorCount)/10
    print("在10次实验过程中，测试集上的平均错误率为%f" % (sumErrorRate / float(10)))

if __name__ == "__main__":
    main()




