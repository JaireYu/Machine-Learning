from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def LoadDataSet(filename):
    fr = open(filename)
    DataMat = []
    for line in fr.readlines():
        Data = line.strip().split('\t')
        line = list(map(float, Data))  #可迭代对象的批操作
        DataMat.append(line)
    print(DataMat)
    return DataMat

def distOhm(vectA, vectB):  #返回向量间的欧氏距离
    return sqrt(sum(power(vectA - vectB, 2)))

def RandCenter(DataSet, k): #生成k个数据域中的随机点作为中心点
    n = shape(DataSet)[1]
    RandCent = mat(zeros((k, n), dtype=float))
    for i in range(n):
        Min = min(DataSet[:, i])
        print(Min)
        Range = float(max(DataSet[:, i]) - Min)
        RandCent[:,i] = Min + Range*random.rand(k, 1)
    return RandCent

def kMeans(DataSet, k, DistMeas = distOhm, CreateCent = RandCenter):
    m = shape(DataSet)[0]   #获取数据个数
    ClusterState = mat(zeros((m, 2))) #用来储存每一个数据点的信息
    CenterMat = CreateCent(DataSet, k)
    ClusterChanging = True
    counter = 1
    while ClusterChanging:
        ClusterChanging = False
        for i in range(m):  #对每一个数据点求最小距离和索引
            MinDist = inf
            MinIndex = -1
            for j in range(k):
                DistI = DistMeas(CenterMat[j], DataSet[i])
                if DistI < MinDist:
                    MinDist = DistI
                    MinIndex = j
            if ClusterState[i,0] != MinIndex:  #如果任意一个数据的最近中心的簇号有更新就进入下一轮
                ClusterChanging = True
            ClusterState[i] = MinIndex, MinDist
        print(CenterMat)
        for center in range(k):
            DataSelect = DataSet[nonzero(ClusterState[:,0].A == center)[0]]    #.A变成array格式，返回m*1的array，nonzero返回按行列的列表*2，选按行的[0]
            CenterMat[center, :] = mean(DataSelect, axis=0)
        Plot(counter, ClusterState, CenterMat, DataMat)
        counter+=1
    plt.show()
    return CenterMat, ClusterState

def Plot(counter, ClusterState, CenterMat, DataMat):
    color = ["red", "purple", "green", "orange"]
    shape = ["+", "+", "+", "+"]
    Position = ["231", "232", "233", "234", "235", "236"]
    plt.suptitle("The process of clustering")
    ax1 = plt.subplot(Position[counter-1])
    plt.title(str(counter))
    for i in range(4):
        Data = DataMat[nonzero(ClusterState[:, 0].A == i)[0]]
        ax1.scatter(Data[:, 0].A.flatten(), Data[:, 1].A.flatten(), color=color[i], alpha=0.6)
        ax1.scatter(CenterMat[i, 0], CenterMat[i, 1], marker=shape[i], color = color[i], s = 80)

if __name__ == "__main__":
    DataMat = mat(LoadDataSet("testSet.txt"))
    CenterMat, ClusterState = kMeans(DataMat, 4)

