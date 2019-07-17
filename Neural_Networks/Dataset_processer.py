from os import listdir
import numpy as np

def img2vector(filename):
    Vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            Vec[0, 32*i + j] = int(lineStr[j])
    return Vec

def GetDataSet():
    FileList = listdir("D:\\python_learning\\KNN_learn\\trainingDigits")
    m = len(FileList)
    Labels = np.zeros((m, 10))
    TrainMat = np.zeros((m, 1024))
    for i in range(m):
        FileNameStr = FileList[i]
        FileStr = FileNameStr.split('.')[0]
        ClassNum = int(FileStr.split('_')[0])
        Labels[i][ClassNum] = 1
        TrainMat[i] = img2vector("D:\\python_learning\\KNN_learn\\trainingDigits\\{}".format(FileNameStr))
    return TrainMat, Labels, m

def GetTestDataSet():
    FileList = listdir("D:\\python_learning\\KNN_learn\\testDigits")
    m = len(FileList)
    Labels = np.zeros((m, 10))
    TrainMat = np.zeros((m, 1024))
    for i in range(m):
        FileNameStr = FileList[i]
        FileStr = FileNameStr.split('.')[0]
        ClassNum = int(FileStr.split('_')[0])
        Labels[i][ClassNum] = 1
        TrainMat[i] = img2vector("D:\\python_learning\\KNN_learn\\testDigits\\{}".format(FileNameStr))
    return TrainMat, Labels, m