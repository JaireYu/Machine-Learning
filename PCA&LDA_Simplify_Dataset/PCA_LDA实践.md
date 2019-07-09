## PCA与LDA实践
### 目标
将iris数据集中的数据从4维降到2维, 并对其可视化
**PS: 理论部分请参见linear_model的LDA部分和PCA&LDA_Simplify_Dataset的数学解释**

### LDA
#### numpy自造模型:
暂无, 待补充
#### sklearn库:
```py
#coding=utf-8
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

colors = ['navy', 'turquoise', 'darkorange']

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):      #将三元数组捆绑成zip遍历三次即可
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)  #这里是选择y==i的行的第0列相当于一种切片方式
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
```
### PCA
#### numpy自造模型
```py
#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets

def pca(dataMat, topN):
    MeanVals = mean(dataMat, axis=0)            #对列求均值
    DeMeanMat = dataMat - MeanVals              #去均值化
    CovMat = cov(DeMeanMat, rowvar= False)      #以列为数据计算协方差矩阵
    EigVals, EigVects = linalg.eig(mat(CovMat)) #求特征值特征向量
    EigValsIndex = argsort(EigVals)             #返回由小到大的索引值
    EigValsIndex = EigValsIndex[:-(topN + 1):-1]#取最大的TopN个
    UsedEigVects = EigVects[:,EigValsIndex]     #选择对应的列(特征向量
    LowDDataMat = DeMeanMat * UsedEigVects      #XE
    ReorganDataMat = LowDDataMat * UsedEigVects.T + MeanVals  #如果正确的话就是DataMat
    return LowDDataMat, ReorganDataMat
```
#### Sklearn_PCA实现:
```py
pca_sklearn = PCA(n_components=2)
X_r = pca_sklearn.fit(X).transform(X)
```
#### 结果对比
```py
#coding=utf-8
import PCA_myway
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

Dimen2Data, OriginalData = PCA_myway.pca(X, 2)

fig = plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
# 我的模型
Dimen2Data = Dimen2Data.getA()
ColorForMe = []
for i in y:
    ColorForMe.append(colors[i])
print(Dimen2Data)
ax1 = fig.add_subplot(211)

ax1.scatter(Dimen2Data[:,0], Dimen2Data[:,1], color = ColorForMe, alpha=.8, lw=lw)
ax1.legend(loc='best', shadow=False, scatterpoints=1)
ax1.set_title('My PCA of IRIS dataset')
#sklearn pca

pca_sklearn = PCA(n_components=2)
X_r = pca_sklearn.fit(X).transform(X)

ax2 = fig.add_subplot(212)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
ax2.legend(loc='best', shadow=False, scatterpoints=1)
ax2.set_title('sklearn PCA of IRIS dataset')

plt.show()
```
**PS: 结果见图片**

### 总结
  **1.numpy和plt的相关用法**
* 添加子图: 
  ```py
  fig = plt.figure()
  ax1 = fig.add_subplot(211) #这里的211是2*1的布局选择第1部分
  ```
* 给子图添加标题:
  ```py
  ax1.set_title("your title")
  ```
* 按索引切片:
  ```py
  array[:,indexlist] #indexlist是索引列表, 表达式返回的是索引列表中的元素对应的列构成的新数组(2X2)
  array[indexlist] #按照索引列表返回行向量构成的数组
  array[:,0] #取二维数组的第0列构成list
  # 对于高维数组可以用array[:,:, num_or_indexlist]表示
  ```
* 按条件切片:
  ```py
  X[y == i]X和y等长的返回的是X中满足条件的相应索引的X行分量构成的数组
  X[y == i, 0]返回的是上面数组的第0列
  X[y == i][0]返回的是上面数组的第0行
  ```
* numpy矩阵运算方法:
  ```py
  1. mean(dataMat, axis=0) #axis=0时返回列均值列表(因为list的大小与axis=0方向相等)
  2. cov(DeMeanMat, rowvar= False)      #计算列间协方差矩阵 否则rowvar = True
  3. EigVals, EigVects = linalg.eig(mat(CovMat)) #求特征值特征向量
   ```
* numpy返回下标的排序方法
  ```py
  EigValsIndex = argsort(EigVals)             #返回由小到大的索引值
  ```
**2.效果**
    在这种有标签的情况下LDA的表现更好, 因为考虑的综合全面, LDA需要Xy, 而PCA只需要X