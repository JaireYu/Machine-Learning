# 通过阈值比较进行分类
from numpy import *
from sklearn import datasets
def StumpClassify(DataMatix, Dimen, ThreshVal, ThreshIneq): #按照阈值和数据维度进行数据分类
    RetArray = ones((shape(DataMatix)[0],1))    #第一项时数据个数
    if ThreshIneq == "lt":
        RetArray[DataMatix[:, Dimen] <= ThreshVal] = -1.0   #当为"lt"是将Dimen列小于Val的索引位置置为-1.0
    else:
        RetArray[DataMatix[:, Dimen] > ThreshVal] = -1.0  # 当为"lt"是将Dimen列小于Val的索引位置置为-1.0
    return RetArray

def BuildStump(DataArr, ClassLabels, D):    #构造单层决策树
    DataMatrix = mat(DataArr)
    LabelMat = mat(ClassLabels).T
    m, n = shape(DataMatrix)
    NumSteps = 10.0; BestStump = {}; BestClassEst = mat(zeros((m, 1)))  #n是单数据属性个数， m是数据个数, BestStump储存最佳决策树参数， BestClassEst储存最佳预测值
    MinError = inf
    for i in range(n):  #对每一个数据维度尝试分类
        RangeMin = DataMatrix[:, i].min()
        RangeMax = DataMatrix[:, i].max()
        StepSize = (RangeMax - RangeMin)/NumSteps   #计算每一次调整阈值的步长
        for j in range(-1, int(NumSteps) + 1):  #调整十次阈值
            for Inequal in ["lt", "gt"]:    #调整预测的正反向
                ThreshVal = (RangeMin + float(j)*StepSize)
                PredictVals = StumpClassify(DataMatrix, i, ThreshVal, Inequal)
                ErrArr = mat(ones((m, 1), dtype=int))
                ErrArr[PredictVals == LabelMat] = 0 #将正确项标注为0
                WeightedError = D.T*ErrArr  #计算加权错误总数
                #print("The WeightedError = {} in condition that ThreshVal = {}, Inequal = {} of the {} dimen\n".format(WeightedError, ThreshVal, Inequal, i))
                if WeightedError < MinError:    #如果最小错误更新字典
                    MinError = WeightedError
                    BestClassEst = PredictVals.copy()
                    BestStump["Dimen"] = i
                    BestStump["Thresh"] = ThreshVal
                    BestStump["Ineq"] = Inequal
    return BestStump, MinError, BestClassEst

def AdaBoostTrain(DataArr, ClassLabels, NumIt = 40):
    WeakClassArr = []
    m = shape(DataArr)[0]
    D = mat(ones(shape = (m, 1))/m)
    AggClassEst = mat(zeros((m, 1)))
    for i in range(NumIt):
        BestStump, Error, ClassEst = BuildStump(DataArr, ClassLabels, D)
        #print("D:", D.T, "\n")
        Alpha = float(0.5*log((1.0-Error)/max(Error, 1e-16)))
        BestStump["alpha"] = Alpha
        WeakClassArr.append(BestStump)  #生成当前层的决策树，将当前层的决策树加入到字典中
        #print("ClassEst:", ClassEst, "\n")
        Expon = multiply(-1*Alpha*mat(ClassLabels).T, ClassEst) #通过预测的准确与否来确定更新系数
        D = multiply(D, exp(Expon)) #更新D矩阵
        D = D/D.sum()
        AggClassEst += Alpha*ClassEst   #每一次生成新的学习器后，都加权加入总的决策
        #print("AggClassEnt", AggClassEst, "\n")
        AggErrors = multiply(sign(AggClassEst)!=mat(ClassLabels).T, ones((m, 1))) #!!!
        ErrorRate = AggErrors.sum()/m
        print("Total errorrate", ErrorRate, "\n")
        if(ErrorRate == 0.0):
            break
    return WeakClassArr

def AdaClassfy(DataToClass, ClassifierArr): #这里Classifier是存储学习器的字典
    DataMat = mat(DataToClass)
    m = shape(DataMat)[0]
    AggClassEst = mat(zeros((m, 1)))
    for i in range(len(ClassifierArr)):
        ClassEst = StumpClassify(DataMat, ClassifierArr[i]["Dimen"],
                                 ClassifierArr[i]["Thresh"], ClassifierArr[i]["Ineq"])
        AggClassEst += ClassifierArr[i]["alpha"]*ClassEst
        #print(AggClassEst)
    return sign(AggClassEst)

def LoadDataSet(filename):
    DataMat = []
    LabelMat = []
    fr = open(filename)
    NumOneLine = len(fr.readline().split('\t'))
    for line in fr.readlines():
        LineInfo = []
        Curline = line.strip().split('\t')
        for i in range(NumOneLine - 1):
            LineInfo.append(float(Curline[i]))
        DataMat.append(LineInfo)
        LabelMat.append(float(Curline[-1]))
    return DataMat, LabelMat

if __name__ == "__main__":
    DataMat, LabelMat = LoadDataSet("HorseTraining.txt")
    DataToClass, RealRes = LoadDataSet("HorseTest.txt")
    AdaBoostClassifier = AdaBoostTrain(DataMat, LabelMat, 50)
    PredictMat = list(AdaClassfy(DataToClass, AdaBoostClassifier))
    Size = len(RealRes)
    count = 0
    for i in range(Size):
        if(RealRes[i] != PredictMat[i]):
            count += 1
    print("The error rate is", count, float(count)/Size)






