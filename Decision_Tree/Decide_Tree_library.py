#coding=utf-8
from math import log
import operator

def CalShannonEntropy(DataSet):     #计算香农熵的函数
    TotalNum = len(DataSet)    #计算数据集中所有元素的个数
    LabelDic = {}    #建立标签的初始字典
    for Data in DataSet:
        CurLabel = Data[-1]
        if CurLabel not in LabelDic.keys():    #如果标签不在字典中
            LabelDic[CurLabel] = 1
        else:   #标签在字典中
            LabelDic[CurLabel] += 1
    Entropy = 0.0
    for elem in LabelDic:
        p = float(LabelDic[elem])/TotalNum
        Entropy = Entropy - p * log(p, 2)   #求熵
    return Entropy

def SpiltDataSet(DataSet, Axis, Value):
    RetDataSet = []
    for Data in DataSet:
        if Data[Axis] == Value:     #如果按Axis号特征是不是Value划分
            ReducedFeat = Data[:Axis]
            ReducedFeat.extend(Data[Axis + 1: ])   #将两片切片和起来得到一个少一个特征的向量
            RetDataSet.append(ReducedFeat)  #将处理后的符合条件的数据向量加入retDataset中
    return RetDataSet

def ChooseBestFeatureToSplit(DataSet):
    NumFeature = len(DataSet[0]) - 1
    BaseEntropy = CalShannonEntropy(DataSet)
    BestFeature = -1
    BestInfoGain = 0.0
    for i in range(NumFeature):
        FeatureList = [example[i] for example in DataSet]
        UniqueValue = set(FeatureList)       #将第i个的所有特征值存在一个集合中
        NewEntropy = 0.0
        for Value in UniqueValue:
            SubDataSet = SpiltDataSet(DataSet, i, Value)    #对于一个特定的Value值求出子集的信息熵
            p = len(SubDataSet)/float(len(DataSet))     #每一个条件的权值
            NewEntropy += p * CalShannonEntropy(SubDataSet)    #加和
        InfoGain = BaseEntropy - NewEntropy     #计算信息增益
        if(InfoGain > BestInfoGain):
            BestInfoGain = InfoGain
            BestFeature = i     #选出信息增益最大的特征序号(即在该特征固定时，会使数据集的熵下降最多
    return BestFeature

def MajorityCnt(ClassList):     #返回ClassList中出现次数最多的特征
    ClassCnt = {}
    for Vote in ClassList:      #遍历ClassList
        if Vote not in ClassCnt.keys(): #如果特征在ClassCnt中没有出现就加到字典中并将值置为0
            ClassCnt[Vote] = 0
        ClassCnt[Vote] += 1
    SortedClassCnt = sorted(ClassCnt.iteritems(), key = operator.itemgetter(1), reverse = True) #第一项是一个迭代器，第二项指出排序的键，是第二项，降序
    return SortedClassCnt[0][0] #返回出现次数最多的特征

def createTree(dataset, labels):    #参数为数据集和标签，创建决策树
    ClassList = [example[-1] for example in dataset]    #选择数据集的最后一项做成ClassList
    if ClassList.count(ClassList[0]) == len(ClassList): #ClassList中的标签完全相同
        return ClassList[0]
    if len(dataset[0]) == 1:    #递归终止条件，遍历完所有特征了，返回出现次数最多的特征
        return MajorityCnt(ClassList)
    BestFeat = ChooseBestFeatureToSplit(dataset)    #在数据集中寻找最佳特征
    BestFeatLabel = labels[BestFeat]    #以特征序号为索引查找Label
    MyTree = {BestFeatLabel: {}}    #MyTree每一个节点上是一个字典，字典的键是当前最佳分类标签，值是结论或者下面的节点
    del(labels[BestFeat])   #从label中删去特征标签
    FeatValues = [example[BestFeat] for example in dataset]     #在dataset里面找该分类标签的所有可能取值
    UniqueVals = set(FeatValues)    #做成集合
    for value in UniqueVals:    #对于该节点的每一个独特的取值都要递归建立子节点
        subLabels = labels[:]
        MyTree[BestFeatLabel][value] = createTree(SpiltDataSet(dataset, BestFeat, value), subLabels)    #得到每一个独特取值对应的子数据集
        #注意：Python的函数参数是列表类型时是以引用的方式传递的，所以要复刻一个sublabel防止label被修改
    return MyTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    SecondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #第一个匹配项的索引位置
    for key in SecondDict.keys():
        if(testVec[featIndex] == key):
            if(type(SecondDict[key]).__name__ == "dict"):
                classLabel = classify(SecondDict[key], featLabels, testVec)
            else:
                classLabel = SecondDict[key]    #如果不是字典就返回分类分支对应的叶子
    return classLabel

def storeTree(inputTree, filename): #用pickle模块将信息写在磁盘上
    import pickle
    fw = open(filename, "w")
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename): #从磁盘反pickle序列化
    import pickle
    fr = open(filename)
    return pickle.load(fr)

