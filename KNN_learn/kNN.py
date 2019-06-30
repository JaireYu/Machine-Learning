#coding=utf-8
from numpy import *
import operator
from os import listdir

def createDataSet(): #制造数据集
    group = array([[1.0,1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k): #根据inX和训练数据集返回inx的预测标签
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile扩展成dataSetSIZE行，1列的
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #沿1轴相加，即二级索引将一个大数组下的小数组的所有元素相加
    distances = sqDistances ** 0.5  #计算距离
    sortedDisIndicies = distances.argsort()
    classCount = {}  #字典
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #获取键对应的值时防止不存在，默认值为0
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0] #返回频率最高的标签

def file2matrix(filename):  #将文件转化为数组
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  #将前三项赋值给returnMat的切片 [index, :]是一个denumpy操作，将numpy矩阵的index行化为普通列表
        classLabelVector.append(int(listFromLine[-1]))  #添加标签
        index = index + 1
    return returnMat, classLabelVector

def autoNorm(dataSet):  #训练数据归一化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))  #每个元素与最小值做差
    normDataSet = normDataSet/tile(ranges, (m,1)) #归一化
    return normDataSet, ranges, minVals

def datingClassTest(): #约会网站测试函数
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('D:\\python_learning\\KNN_learn\\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs): #numTestVecs是测试数据，剩下m-numTestVecs作为"训练数据"
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d", classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print("the total error rate is %f", errorCount/float(numTestVecs))

def img2vector(filename):   #将手写的数字矩阵转化成
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir("D:\\python_learning\\KNN_learn\\trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #取文件名去掉扩展名
        classNum =  int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector("D:\\python_learning\\KNN_learn\\trainingDigits\\{}".format(fileNameStr))
    testFileList = listdir("D:\\python_learning\\KNN_learn\\testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 取文件名去掉扩展名
        classNum = int(fileStr.split('_')[0])
        vectorTest = img2vector("D:\\python_learning\\KNN_learn\\testDigits\\{}".format(fileNameStr))
        classifierResult = classify0(vectorTest, trainingMat, hwLabels, 3)
        print("the classifier come back with {} while the real number is {}".format(classifierResult, classNum))
        if(classNum != classifierResult):
            errorCount += 1.0
    print("\nthe total error rate is {}".format(errorCount/float(mTest)))

def hwTest(filename):
    hwLabels = []
    trainingFileList = listdir("D:\\python_learning\\KNN_learn\\trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector("D:\\python_learning\\KNN_learn\\trainingDigits\\{}".format(fileNameStr))
    vectorTest = img2vector(filename)
    Result = classify0(vectorTest, trainingMat, hwLabels, 5)
    print("The handwriting is ", Result)



